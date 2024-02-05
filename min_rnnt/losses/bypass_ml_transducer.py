from contextlib import nullcontext

import k2
import torch
import torch.nn.functional as F
from nemo.collections.asr.parts.k2.graph_transducer import GraphRnntLoss, force_float32_context


class GraphBypassMultiLevelTransducerLoss(GraphRnntLoss):
    def __init__(
        self,
        blank: int,
        drop_prob: float = 0.2,
        skip_token_penalty: float = 0.0,
        skip_token_mode: str = "mean",
        return_graph: bool = False,
        use_grid_implementation=False,  # TODO: grid impl
        connect_composed=False,
        double_scores=False,
        cast_to_float32=False,
    ):
        super().__init__(
            blank=blank,
            use_grid_implementation=use_grid_implementation,
            connect_composed=connect_composed,
            double_scores=double_scores,
            cast_to_float32=cast_to_float32,
        )
        self.drop_prob = drop_prob
        self.return_graph = return_graph
        self.skip_token_penalty = skip_token_penalty
        self.skip_token_mode = skip_token_mode

    def get_unit_schema(self, units_tensor: torch.Tensor, vocab_size: int) -> "k2.Fsa":
        """
        Get unit schema (target text) graph for Bypass-Transducer loss (Compose-Transducer).
        Forward arcs represent text labels.

        Example graph: text [1, 2], blank=0. Eps id: 3.

        graph::

                0:0:0                  0:0:1                  0:0:2
              +-------+              +-------+              +-------+
              v       |              v       |              v       |
            +-----------+  1:1:0   +-----------+  2:2:1   +-----------+  -1:-1:-1  #===#
            |     0     | -------> |     1     | -------> |     2     | ---------> H 3 H
            +-----------+ -------> +-----------+          +-----------+            #===#

        Args:
            units_tensor: 1d tensor with text units
            vocab_size: number of total labels (vocab size including blank)

        Returns:
            unit schema graph (k2.Fsa).
            Labels: <unit>:<unit>:<unit_position> (k2.Fsa: labels, aux_labels, unit_positions)
        """

        blank_id = self.blank
        skip_token_id = vocab_size
        device = units_tensor.device
        text_len = units_tensor.shape[0]
        units = units_tensor.tolist()
        num_levels = text_len * self.drop_prob
        arcs = []
        last_state = num_levels * (text_len + 1) * 3 + 1
        state = 0
        for level in range(num_levels):
            for i in range(level, text_len + 1):
                # self-loop
                arcs.append([state, state, blank_id, blank_id, i, 0])
                # forward
                if i < text_len:
                    arcs.append([state, state + 1, units[i], units[i], i, 0])
                    # skip
                    if level < num_levels - 1:
                        arcs.append([state, state + text_len - level + 1, skip_token_id, skip_token_id, i, 0])
                else:
                    arcs.append([state, last_state, -1, -1, -1, 0])
                state += 1
        arcs.append([last_state])
        schema_fst_str = "\n".join([" ".join(map(str, line)) for line in arcs])
        fsa_text = k2.Fsa.from_str(schema_fst_str, aux_label_names=["aux_labels", "unit_positions"]).to(device)
        return fsa_text

    def get_temporal_schema(self, num_frames: int, vocab_size: int, device: torch.device) -> "k2.Fsa":
        """
        Get temporal schema graph for Star-Transducer loss (Compose-Transducer).

        Example graph: blank=0, num_frames=3, vocab_size=3, last_blank_mode="force_final".
        Labels: <unit>:<frame_index>. <unit> is a unit from vocab + special eps ids `vocab_size`.


        Args:
            num_frames: length of the sequence (in frames)
            vocab_size: number of labels (including blank)
            device: device for tensor to construct

        Returns:
            temporal schema graph (k2.Fsa).
            Labels: <unit>:<frame_index>. <unit> is a unit from vocab + special units (e.g., additional eps).
        """
        blank_id = self.blank

        fsa_temporal_arcs = torch.zeros((num_frames * (vocab_size + 1) + 1, 4), dtype=torch.int32, device=device)
        sequence_states = torch.arange(0, num_frames, dtype=torch.int32, device=device)
        # for every state - vocab_size arcs, [0, 1, ..., vocab_size-1, 0, 1, ..., vocab_size-1, ...]
        start_states = sequence_states.expand(vocab_size + 1, num_frames).transpose(0, 1).flatten()
        # first: make all arcs - self-loops
        fsa_temporal_arcs[:-1, 0] = start_states  # from
        fsa_temporal_arcs[:-1, 1] = start_states  # to
        fsa_temporal_arcs[:-1, 2] = (
            torch.arange(0, vocab_size + 1, dtype=torch.int32, device=device)
            .expand(num_frames, vocab_size + 1)
            .flatten()
        )

        # blank-arcs: forward
        fsa_temporal_arcs[blank_id : -1 : vocab_size + 1, 1] = sequence_states + 1  # blanks

        # transition to last final state
        fsa_temporal_arcs[-1, :3] = torch.tensor((num_frames, num_frames + 1, -1), dtype=torch.int32, device=device)

        # output symbols: position in the sequence, same as start states for arcs
        olabels = fsa_temporal_arcs[:, 0].detach().clone()
        olabels[-1] = -1  # last arc to final state

        fsa_temporal = k2.Fsa(fsa_temporal_arcs, olabels)
        fsa_temporal = k2.arc_sort(fsa_temporal)  # need for compose
        return fsa_temporal

    def get_grid(self, units_tensor: torch.Tensor, num_frames: int, vocab_size: int) -> "k2.Fsa":
        raise NotImplementedError

    def forward(
        self,
        acts: torch.Tensor,
        labels: torch.Tensor,
        act_lens: torch.Tensor,
        label_lens: torch.Tensor,
    ):
        """
        Forward method is similar to RNN-T Graph-Transducer forward method,
        but we need to assign eps weight to eps-transitions.
        """
        # argument names are consistent with NeMo, see RNNTLoss.forward:
        # self._loss(acts=log_probs, labels=targets, act_lens=input_lengths, label_lens=target_lengths)
        logits, targets, logits_lengths, target_lengths = acts, labels, act_lens, label_lens

        # logits: B x Time x Text+1 x C
        vocab_size = logits.shape[-1]
        batch_size = logits.shape[0]
        target_fsas_vec = self.get_graphs_batched(logits_lengths, targets, target_lengths, vocab_size)

        cast_context = force_float32_context() if self.cast_to_float32 else nullcontext()
        with cast_context:
            # activation: log softmax
            log_probs = F.log_softmax(logits, dim=-1)
            with torch.no_grad():
                last_transition_mask = target_fsas_vec.labels == -1
                skip_token_transition_mask = target_fsas_vec.labels == vocab_size

                batch_indices = last_transition_mask.cumsum(dim=-1) - last_transition_mask.to(torch.long)
                time_indices = target_fsas_vec.aux_labels.clone().to(torch.int64)
                unit_indices = target_fsas_vec.unit_positions.clone().to(torch.int64)
                text_units = target_fsas_vec.labels.clone().to(torch.int64)

                # eps transitions
                text_units.masked_fill_(last_transition_mask, 0)

                # skip frames transitions
                text_units.masked_fill_(skip_token_transition_mask, 0)

            # NB: do not assign scores -> modify, k2 will not update all scores correctly (modify -> assign)
            scores = log_probs[batch_indices, time_indices, unit_indices, text_units]
            # fix weights for the arcs to the last state
            scores[last_transition_mask] = 0

            # assign skip_frame penalty to skip_frame arcs
            assert self.blank == vocab_size - 1
            match self.skip_token_mode:
                case "constant":
                    scores[skip_token_transition_mask] = self.skip_token_penalty
                case "mean":
                    # similar to OTC implemenetation: https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/WSASR/conformer_ctc2/train.py#L568
                    mean_logprob = torch.logsumexp(log_probs[..., : self.blank], dim=-1, keepdim=False) - torch.log(
                        torch.full([batch_size], fill_value=vocab_size - 1, device=log_probs.device)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    mean_scores = mean_logprob[batch_indices, time_indices, unit_indices]
                    scores = torch.where(skip_token_transition_mask, mean_scores, scores)
                    scores[skip_token_transition_mask] += self.skip_token_penalty
                case "max":
                    max_logprob, _ = log_probs[..., : self.blank].max(dim=-1, keepdim=False)
                    max_scores = max_logprob[batch_indices, time_indices, unit_indices]
                    scores = torch.where(skip_token_transition_mask, max_scores, scores)
                    scores[skip_token_transition_mask] += self.skip_token_penalty
                case "maxexcl":
                    device = log_probs.device
                    max_text_len = log_probs.shape[2] - 1
                    # assert max_text_len == target_lengths.max()
                    batch_indices_f = (
                        torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_text_len).flatten()
                    )
                    unit_position_indices_f = (
                        torch.arange(max_text_len, device=device)
                        .unsqueeze(0)
                        .expand(batch_size, max_text_len)
                        .flatten()
                    )
                    text_units_f = targets.flatten()
                    log_probs_modified = log_probs.clone()
                    log_probs_modified[batch_indices_f, :, unit_position_indices_f, text_units_f] = float("-inf")
                    log_probs_modified[
                        batch_indices_f,
                        :,
                        unit_position_indices_f,
                        torch.full_like(text_units_f, fill_value=self.blank),
                    ] = float("-inf")
                    max_logprob, _ = log_probs_modified[..., : self.blank].max(dim=-1, keepdim=False)
                    # print(max_logprob)
                    max_scores = max_logprob[batch_indices, time_indices, unit_indices]
                    # print(scores, max_scores)
                    scores = torch.where(skip_token_transition_mask, max_scores, scores)
                    scores[skip_token_transition_mask] += self.skip_token_penalty
                case "sumexcl":
                    device = log_probs.device
                    max_text_len = log_probs.shape[2] - 1
                    # assert max_text_len == target_lengths.max()
                    batch_indices_f = (
                        torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_text_len).flatten()
                    )
                    unit_position_indices_f = (
                        torch.arange(max_text_len, device=device)
                        .unsqueeze(0)
                        .expand(batch_size, max_text_len)
                        .flatten()
                    )
                    text_units_f = targets.flatten()
                    log_probs_modified = log_probs.clone()
                    log_probs_modified[batch_indices_f, :, unit_position_indices_f, text_units_f] = float("-inf")
                    log_probs_modified[
                        batch_indices_f,
                        :,
                        unit_position_indices_f,
                        torch.full_like(text_units_f, fill_value=self.blank),
                    ] = float("-inf")
                    sum_logprobs = torch.logsumexp(log_probs_modified[..., : self.blank], dim=-1, keepdim=False)
                    # print(max_logprob)
                    sum_scores = sum_logprobs[batch_indices, time_indices, unit_indices]
                    # print(scores, max_scores)
                    scores = torch.where(skip_token_transition_mask, sum_scores, scores)
                    scores[skip_token_transition_mask] += self.skip_token_penalty
                case _:
                    raise NotImplementedError

            target_fsas_vec.scores = scores

            # compute loss: full-sum
            scores = -1 * target_fsas_vec.get_tot_scores(use_double_scores=self.double_scores, log_semiring=True)
            if self.return_graph:
                return scores, target_fsas_vec
            return scores
