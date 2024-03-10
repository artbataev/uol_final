# This is the part of the Final Project
# "RNN-Transducer-based Losses for Speech Recognition on Noisy Targets"
# Originally published in https://github.com/artbataev/uol_final

from contextlib import nullcontext

import k2
import torch
import torch.nn.functional as F
from nemo.collections.asr.parts.k2.graph_transducer import GraphRnntLoss, force_float32_context


class GraphTargetRobustTransducerLoss(GraphRnntLoss):
    """
    Original implementation of Target-Robust Transducer loss.
    The implementation is based on "Graph-based framework  for RNN-Transducer losses",
    For the original implementation see
    https://github.com/NVIDIA/NeMo/blob/v1.21.0/nemo/collections/asr/parts/k2/graph_transducer.py

    This loss it the combination of Bypass Transducer and Star Transducer losses:  we augment the RNN-Transducer
    with additional skip_frame arcs parallel to blank arcs and skip_token arcs parallel to text arcs.
    The loss is useful for training RNN-T system with partially correct transcripts
    (with substitutions or arbitrary errors).
    """

    def __init__(
        self,
        blank: int,
        skip_frame_penalty: float = 0.0,
        skip_token_penalty: float = 0.0,
        skip_token_mode: str = "sumexcl",
        use_grid_implementation=True,
        connect_composed=False,
        double_scores=False,
        cast_to_float32=False,
    ):
        """
        Init method

        :param blank: blank label index
        :param skip_frame_penalty: weight of epsilon transitions, 0 means no penalty (default)
        :param skip_token_penalty: weight of skip token transition, 0 means no penalty
        :param skip_token_mode: mode to assign weight to skip token transition,
                options "sumexcl" (default, found to be best), "maxexcl", "sum", "mean", "const"
        :param use_grid_implementation: Whether to use the grid implementation (Grid-Transducer).
        :param connect_composed: Connect graph after composing unit and temporal schemas
                (only for Compose-Transducer). `connect` operation is slow, it is useful for visualization,
                but not necessary for loss computation.
        :param double_scores: Use calculation of loss in double precision (float64) in the lattice.
            Does not significantly affect memory usage since the lattice is ~V/2 times smaller than the joint tensor.
        :param cast_to_float32: Force cast joint tensor to float32 before log-softmax calculation.
        """
        super().__init__(
            blank=blank,
            use_grid_implementation=use_grid_implementation,
            connect_composed=connect_composed,
            double_scores=double_scores,
            cast_to_float32=cast_to_float32,
        )
        self.skip_frame_penalty = skip_frame_penalty
        self.skip_token_penalty = skip_token_penalty
        self.skip_frame_id_rel = 0  # skip_frame_id = vocab_size + 0
        self.skip_token_id_rel = 1  # skip_token_id = vocab_size + 1
        self.skip_token_mode = skip_token_mode

    def get_unit_schema(self, units_tensor: torch.Tensor, vocab_size: int) -> "k2.Fsa":
        """
        Get unit schema (target text) graph for Star-Transducer loss (Compose-Transducer).
        We add both skip token (from Bypass-Transducer) and skip frame (from Star-Transducer) transitions
        """
        blank_id = self.blank
        skip_frame_id = vocab_size + self.skip_frame_id_rel
        skip_token_id = vocab_size + self.skip_token_id_rel
        device = units_tensor.device
        text_len = units_tensor.shape[0]

        # units_transitions: scr, dest, label, score (k2 convention)
        units_transitions = torch.zeros((text_len * 4 + 2 + 1, 4), dtype=torch.int32, device=device)
        text_indices = torch.arange(0, text_len + 1, dtype=torch.int32, device=device)
        # fill blank labels: self-loops, each 4th element
        units_transitions[0:-1:4, 0] = text_indices  # from state
        units_transitions[0:-1:4, 1] = text_indices  # to state
        units_transitions[0:-1:4, 2] = blank_id

        # skip frame labels - self-loops
        units_transitions[1:-1:4, 0] = text_indices  # from state
        units_transitions[1:-1:4, 1] = text_indices  # to state
        units_transitions[1:-1:4, 2] = skip_frame_id

        # text labels - forward
        units_transitions[2::4, 0] = text_indices  # from state
        units_transitions[2::4, 1] = text_indices + 1  # to state (next)
        units_transitions[2:-1:4, 2] = units_tensor  # labels: text

        # skip token labels - parallel to text labels
        units_transitions[3::4, 0] = text_indices[:-1]  # from state
        units_transitions[3::4, 1] = text_indices[:-1] + 1  # to state (next)
        units_transitions[3:-1:4, 2] = skip_token_id  # labels: text

        # last transition to final state, `-1` label (special for k2)
        units_transitions[-1, 2] = -1
        # output labels: same as input labels = text units
        olabels = units_transitions[:, 2].detach().clone()  # same as ilabels

        fsa_units_trt = k2.Fsa(units_transitions, olabels)
        # fill in unit positions
        fsa_units_trt.unit_positions = text_indices.expand(4, -1).transpose(0, 1).flatten()[:-1]
        fsa_units_trt.unit_positions[-1] = -1
        return fsa_units_trt

    def get_temporal_schema(self, num_frames: int, vocab_size: int, device: torch.device) -> "k2.Fsa":
        """
        Get temporal schema graph for Star-Transducer loss (Compose-Transducer).
        Schema is similar to RNN-T, but with additional self-loops for "skip token" and "skip frame" transitions
        """
        blank_id = self.blank

        skip_frame_id = vocab_size + self.skip_frame_id_rel
        skip_token_id = vocab_size + self.skip_token_id_rel

        temporal_transitions = torch.zeros((num_frames * (vocab_size + 2), 4), dtype=torch.int32, device=device)
        sequence_states = torch.arange(0, num_frames, dtype=torch.int32, device=device)
        # for every state - vocab_size arcs, [0, 1, ..., vocab_size-1, 0, 1, ..., vocab_size-1, ...]
        start_states = sequence_states.expand(vocab_size + 2, num_frames).transpose(0, 1).flatten()
        # self-loops: emit all tokens from the vocabulary;
        # then replace transitions to next state when necessary
        temporal_transitions[:, 0] = start_states  # from
        temporal_transitions[:, 1] = start_states  # to
        temporal_transitions[:, 2] = (
            torch.arange(0, vocab_size + 2, dtype=torch.int32, device=device)
            .expand(num_frames, vocab_size + 2)
            .flatten()
        )

        # add blank transitions: forward (by time)
        temporal_transitions[blank_id : -1 : vocab_size + 2, 1] = sequence_states + 1  # blanks
        temporal_transitions[skip_frame_id : -2 : vocab_size + 2, 1] = (sequence_states + 1)[:-1]  # [:-1]  # blanks
        temporal_transitions[-2, 2] = skip_token_id

        # add transition to the last final state (should be separated in k2)
        temporal_transitions[-1, :3] = torch.tensor((num_frames, num_frames + 1, -1), dtype=torch.int32, device=device)

        # output symbols: position in the sequence, same as start states for arcs
        olabels = temporal_transitions[:, 0].detach().clone()
        # special for k2 transition to the last state with label -1
        olabels[-1] = -1  # last arc to final state

        # construct FSA
        fsa_temporal_trt = k2.Fsa(temporal_transitions, olabels)
        # sort arcs - necessary for composition
        fsa_temporal_trt = k2.arc_sort(fsa_temporal_trt)
        return fsa_temporal_trt

    def get_grid(self, units_tensor: torch.Tensor, num_frames: int, vocab_size: int) -> "k2.Fsa":
        """
        Directly construct lattice for Target-Robust-Transducer
        We use grid labeled enumerated by rows/columns
        """
        blank_id = self.blank
        # skip frame and skip token ids - outside vocab
        skip_frame_id = vocab_size + self.skip_frame_id_rel
        skip_token_id = vocab_size + self.skip_token_id_rel

        text_length = units_tensor.shape[0]
        device = units_tensor.device
        num_grid_states = num_frames * (text_length + 1)
        num_blank_arcs = (num_frames - 1) * (text_length + 1)
        num_text_arcs = text_length * num_frames
        transitions = torch.zeros((num_blank_arcs * 2 + num_text_arcs * 2 + 2, 4), dtype=torch.int32, device=device)
        # blank transitions
        from_states = torch.arange(num_blank_arcs, device=device)
        to_states = from_states + (text_length + 1)

        transitions[:num_blank_arcs, 0] = from_states
        transitions[:num_blank_arcs, 1] = to_states
        transitions[:num_blank_arcs, 2] = blank_id

        # skip_frame transitions
        transitions[num_blank_arcs : num_blank_arcs * 2, 0] = from_states
        transitions[num_blank_arcs : num_blank_arcs * 2, 1] = to_states
        transitions[num_blank_arcs : num_blank_arcs * 2, 2] = skip_frame_id

        # text transitions
        from_states = (
            torch.arange(num_grid_states, dtype=torch.int32, device=device)
            .reshape(num_frames, text_length + 1)[:, :-1]
            .flatten()
        )
        to_states = from_states + 1
        ilabels = units_tensor.expand(num_frames, -1).flatten()
        transitions[num_blank_arcs * 2 : num_blank_arcs * 2 + num_text_arcs, 0] = from_states
        transitions[num_blank_arcs * 2 : num_blank_arcs * 2 + num_text_arcs, 1] = to_states
        transitions[num_blank_arcs * 2 : num_blank_arcs * 2 + num_text_arcs, 2] = ilabels

        # skip token arcs - parallel to text transitions
        transitions[num_blank_arcs * 2 + num_text_arcs : -2, 0] = from_states
        transitions[num_blank_arcs * 2 + num_text_arcs : -2, 1] = to_states
        transitions[num_blank_arcs * 2 + num_text_arcs : -2, 2] = skip_token_id

        # last 2 states
        # last blank label
        transitions[-2, :3] = torch.tensor(
            (num_grid_states - 1, num_grid_states, blank_id), dtype=torch.int32, device=device
        )
        # final state
        transitions[-1, :3] = torch.tensor(
            (num_grid_states, num_grid_states + 1, -1), dtype=torch.int32, device=device
        )

        # sequence indices, time indices
        # transitions[:, 0] // (text_length + 1)
        olabels = torch.div(transitions[:, 0], (text_length + 1), rounding_mode="floor")
        unit_positions = transitions[:, 0] % (text_length + 1)
        # last state: final
        olabels[-1] = -1
        unit_positions[-1] = -1

        # relabel states to speedup k2 computations, reusing method from original GraphRnntLoss
        transitions[:-2, 0] = self.relabel_states(transitions[:-2, 0], text_length + 1, num_frames)
        transitions[:-3, 1] = self.relabel_states(transitions[:-3, 1], text_length + 1, num_frames)

        # sort by start state - required in k2
        indices = torch.argsort(transitions[:, 0], dim=0)
        sorted_arcs = transitions[indices]
        olabels = olabels[indices]
        unit_positions = unit_positions[indices]

        # construct lattice
        trt_graph = k2.Fsa(sorted_arcs, olabels)
        trt_graph.unit_positions = unit_positions
        return trt_graph

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

        # logits: Batch x Time x TextUnits+1 x Vocab
        vocab_size = logits.shape[-1]
        batch_size = logits.shape[0]
        device = logits.device

        # construct computational lattices (composed or directly grids)
        # the method is reused from GraphRnntLoss, and retrieves composition or lattice directly
        # for each item in the batch, based on self.use_grid_implementation parameter
        target_fsas_vec = self.get_graphs_batched(logits_lengths, targets, target_lengths, vocab_size)

        skip_frame_id = vocab_size + self.skip_frame_id_rel
        skip_token_id = vocab_size + self.skip_token_id_rel

        cast_context = force_float32_context() if self.cast_to_float32 else nullcontext()
        with cast_context:
            # use activation: log softmax (log probabilities - output of Joint)
            log_probs = F.log_softmax(logits, dim=-1)
            with torch.no_grad():
                # mask for last transition - for all graphs
                last_transition_mask = target_fsas_vec.labels == -1
                # mask for skip token transitions - for all graphs
                skip_token_transition_mask = target_fsas_vec.labels == skip_token_id
                # mask for skip frame transitions - for all graphs
                skip_frame_transition_mask = target_fsas_vec.labels == skip_frame_id

                # batch indices for all graph transitions
                batch_indices = torch.repeat_interleave(
                    torch.arange(batch_size, device=device, dtype=torch.int64),
                    torch.tensor(
                        [target_fsas_vec.arcs.index(0, i)[0].values().shape[0] for i in range(batch_size)],
                        device=device,
                    ),
                )
                time_indices = target_fsas_vec.aux_labels.clone().to(torch.int64)
                unit_indices = target_fsas_vec.unit_positions.clone().to(torch.int64)
                text_units = target_fsas_vec.labels.clone().to(torch.int64)

                # transition to the last state + eps-transitions
                # use 0 index (for valid index_select) and manually assign score after index_select for this case
                # eps transitions
                text_units.masked_fill_(last_transition_mask, 0)
                # skip tokens transitions
                text_units.masked_fill_(skip_token_transition_mask, 0)
                # skip frames transitions
                text_units.masked_fill_(skip_frame_transition_mask, 0)

            # fill in transition scores
            scores = log_probs[batch_indices, time_indices, unit_indices, text_units]
            # fix weights for the transitions to the last state (special for k2)
            scores[last_transition_mask] = 0
            # skip frame penalty - assign constant
            scores[skip_frame_transition_mask] = self.skip_frame_penalty

            assert self.blank == vocab_size - 1  # convention for RNN-T models in NeMo, blank is the last
            # assign skip token penalty, similar to Bypass-Transducer loss
            match self.skip_token_mode:
                case "constant":
                    # simple constant assignment
                    scores[skip_token_transition_mask] = self.skip_token_penalty
                case "mean":
                    # similar to OTC implemenetation:
                    # https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/WSASR/conformer_ctc2/train.py#L568
                    # logsumexp for all logits before blank
                    mean_logprob = torch.logsumexp(log_probs[..., : self.blank], dim=-1, keepdim=False) - torch.log(
                        torch.full([batch_size], fill_value=vocab_size - 1, device=log_probs.device)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    mean_scores = mean_logprob[batch_indices, time_indices, unit_indices]
                    scores = torch.where(skip_token_transition_mask, mean_scores, scores)
                    # add constant penalty
                    scores[skip_token_transition_mask] += self.skip_token_penalty
                case "max":
                    # maximum log probability for all outputs excluding blank
                    max_logprob, _ = log_probs[..., : self.blank].max(dim=-1, keepdim=False)
                    max_scores = max_logprob[batch_indices, time_indices, unit_indices]
                    scores = torch.where(skip_token_transition_mask, max_scores, scores)
                    # add constant penalty
                    scores[skip_token_transition_mask] += self.skip_token_penalty
                case "maxexcl":
                    # maximum log probability for all outputs excluding blank and target token
                    device = log_probs.device
                    max_text_len = log_probs.shape[2] - 1
                    # idea: copy log probs, assign -inf to target and blank labels, get maximum
                    log_probs_modified = log_probs.clone()

                    # indexing for logits
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
                    # assign -inf to target units
                    log_probs_modified[batch_indices_f, :, unit_position_indices_f, text_units_f] = float("-inf")
                    # assign -inf to blank units
                    log_probs_modified[
                        batch_indices_f,
                        :,
                        unit_position_indices_f,
                        torch.full_like(text_units_f, fill_value=self.blank),
                    ] = float("-inf")
                    # get maximum
                    max_logprob, _ = log_probs_modified[..., : self.blank].max(dim=-1, keepdim=False)
                    max_scores = max_logprob[batch_indices, time_indices, unit_indices]
                    scores = torch.where(skip_token_transition_mask, max_scores, scores)
                    # add constant penalty
                    scores[skip_token_transition_mask] += self.skip_token_penalty
                case "sumexcl":
                    device = log_probs.device
                    max_text_len = log_probs.shape[2] - 1

                    # idea: copy log probs, assign -inf to target and blank labels, get logsumexp
                    log_probs_modified = log_probs.clone()
                    # indexing for logits
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
                    # assign -inf to target units
                    log_probs_modified[batch_indices_f, :, unit_position_indices_f, text_units_f] = float("-inf")
                    # assign -inf to blank units
                    log_probs_modified[
                        batch_indices_f,
                        :,
                        unit_position_indices_f,
                        torch.full_like(text_units_f, fill_value=self.blank),
                    ] = float("-inf")
                    # get log sum exp
                    sum_logprobs = torch.logsumexp(log_probs_modified[..., : self.blank], dim=-1, keepdim=False)
                    sum_scores = sum_logprobs[batch_indices, time_indices, unit_indices]
                    scores = torch.where(skip_token_transition_mask, sum_scores, scores)
                    # add constant penalty
                    scores[skip_token_transition_mask] += self.skip_token_penalty
                case _:
                    # reserved for future experiments
                    raise NotImplementedError

            # assign scores to graphs
            target_fsas_vec.scores = scores

            # compute loss: full-sum loss using k2 graph method
            scores = -1 * target_fsas_vec.get_tot_scores(use_double_scores=self.double_scores, log_semiring=True)
            return scores
