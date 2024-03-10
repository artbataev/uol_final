# This is the part of the Final Project
# "RNN-Transducer-based Losses for Speech Recognition on Noisy Targets"
# Originally published in https://github.com/artbataev/uol_final

import math
import random
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
        use_alignment_prob=0.0,
    ):
        """
        Init method

        :param blank: blank label index
        :param skip_frame_penalty: weight of epsilon transitions, 0 means no penalty (default)
        :param skip_token_penalty: weight of skip token transition, 0 means no penalty
        :param skip_token_mode: mode to assign weight to skip token transition,
                options "sumexcl" (default, found to be best), "maxexcl", "meanexcl", "sum", "mean", "const"
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
        self.use_alignment_prob = use_alignment_prob

    def get_unit_schema(self, units_tensor: torch.Tensor, vocab_size: int) -> "k2.Fsa":
        """
        Get unit schema (target text) graph for Star-Transducer loss (Compose-Transducer).
        Forward arcs represent text labels.

        Args:
            units_tensor: 1d tensor with text units
            vocab_size: number of total labels (vocab size including blank)

        Returns:
            unit schema graph (k2.Fsa).
            Labels: <unit>:<unit>:<unit_position> (k2.Fsa: labels, aux_labels, unit_positions)
        """

        blank_id = self.blank
        skip_frame_id = vocab_size + self.skip_frame_id_rel
        skip_token_id = vocab_size + self.skip_token_id_rel
        device = units_tensor.device
        text_len = units_tensor.shape[0]

        # arcs: scr, dest, label, score
        arcs = torch.zeros((text_len * 4 + 2 + 1, 4), dtype=torch.int32, device=device)
        text_indices = torch.arange(0, text_len + 1, dtype=torch.int32, device=device)
        # blank labels
        arcs[0:-1:4, 0] = text_indices  # from state
        arcs[0:-1:4, 1] = text_indices  # to state
        arcs[0:-1:4, 2] = blank_id

        # eps labels
        arcs[1:-1:4, 0] = text_indices  # from state
        arcs[1:-1:4, 1] = text_indices  # to state
        arcs[1:-1:4, 2] = skip_frame_id

        # text labels
        arcs[2::4, 0] = text_indices  # from state
        arcs[2::4, 1] = text_indices + 1  # to state (next)
        arcs[2:-1:4, 2] = units_tensor  # labels: text

        # skip text labels
        arcs[3::4, 0] = text_indices[:-1]  # from state
        arcs[3::4, 1] = text_indices[:-1] + 1  # to state (next)
        arcs[3:-1:4, 2] = skip_token_id  # labels: text

        arcs[-1, 2] = -1  # last transition to final state, ilabel=-1 (special for k2)
        olabels = arcs[:, 2].detach().clone()  # same as ilabels

        fsa_text = k2.Fsa(arcs, olabels)
        fsa_text.unit_positions = text_indices.expand(4, -1).transpose(0, 1).flatten()[:-1]
        fsa_text.unit_positions[-1] = -1
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

        skip_frame_id = vocab_size + self.skip_frame_id_rel
        skip_token_id = vocab_size + self.skip_token_id_rel

        fsa_temporal_arcs = torch.zeros((num_frames * (vocab_size + 2), 4), dtype=torch.int32, device=device)
        sequence_states = torch.arange(0, num_frames, dtype=torch.int32, device=device)
        # for every state - vocab_size arcs, [0, 1, ..., vocab_size-1, 0, 1, ..., vocab_size-1, ...]
        start_states = sequence_states.expand(vocab_size + 2, num_frames).transpose(0, 1).flatten()
        # first: make all arcs - self-loops
        fsa_temporal_arcs[:, 0] = start_states  # from
        fsa_temporal_arcs[:, 1] = start_states  # to
        fsa_temporal_arcs[:, 2] = (
            torch.arange(0, vocab_size + 2, dtype=torch.int32, device=device)
            .expand(num_frames, vocab_size + 2)
            .flatten()
        )

        # blank-arcs: forward
        fsa_temporal_arcs[blank_id : -1 : vocab_size + 2, 1] = sequence_states + 1  # blanks
        fsa_temporal_arcs[skip_frame_id : -2 : vocab_size + 2, 1] = (sequence_states + 1)[:-1]  # [:-1]  # blanks
        fsa_temporal_arcs[-2, 2] = skip_token_id

        # transition to last final state
        fsa_temporal_arcs[-1, :3] = torch.tensor((num_frames, num_frames + 1, -1), dtype=torch.int32, device=device)

        # output symbols: position in the sequence, same as start states for arcs
        olabels = fsa_temporal_arcs[:, 0].detach().clone()
        olabels[-1] = -1  # last arc to final state

        fsa_temporal = k2.Fsa(fsa_temporal_arcs, olabels)
        fsa_temporal = k2.arc_sort(fsa_temporal)  # need for compose
        return fsa_temporal

    def get_grid(self, units_tensor: torch.Tensor, num_frames: int, vocab_size: int) -> "k2.Fsa":
        blank_id = self.blank
        skip_frame_id = vocab_size + self.skip_frame_id_rel
        skip_token_id = vocab_size + self.skip_token_id_rel

        text_length = units_tensor.shape[0]
        device = units_tensor.device
        num_grid_states = num_frames * (text_length + 1)
        num_blank_arcs = (num_frames - 1) * (text_length + 1)
        num_text_arcs = text_length * num_frames
        arcs = torch.zeros((num_blank_arcs * 2 + num_text_arcs * 2 + 2, 4), dtype=torch.int32, device=device)
        # blank transitions
        # i, i+<text_len + 1>, 0 <blank>, i / <text_len+1>, i % <text_len + 1>
        from_states = torch.arange(num_blank_arcs, device=device)
        to_states = from_states + (text_length + 1)

        # blank
        arcs[:num_blank_arcs, 0] = from_states
        arcs[:num_blank_arcs, 1] = to_states
        arcs[:num_blank_arcs, 2] = blank_id

        # skip_frame
        arcs[num_blank_arcs : num_blank_arcs * 2, 0] = from_states
        arcs[num_blank_arcs : num_blank_arcs * 2, 1] = to_states
        arcs[num_blank_arcs : num_blank_arcs * 2, 2] = skip_frame_id

        # text arcs
        from_states = (
            torch.arange(num_grid_states, dtype=torch.int32, device=device)
            .reshape(num_frames, text_length + 1)[:, :-1]
            .flatten()
        )
        to_states = from_states + 1
        ilabels = units_tensor.expand(num_frames, -1).flatten()
        arcs[num_blank_arcs * 2 : num_blank_arcs * 2 + num_text_arcs, 0] = from_states
        arcs[num_blank_arcs * 2 : num_blank_arcs * 2 + num_text_arcs, 1] = to_states
        arcs[num_blank_arcs * 2 : num_blank_arcs * 2 + num_text_arcs, 2] = ilabels

        # skip arcs
        arcs[num_blank_arcs * 2 + num_text_arcs : -2, 0] = from_states
        arcs[num_blank_arcs * 2 + num_text_arcs : -2, 1] = to_states
        arcs[num_blank_arcs * 2 + num_text_arcs : -2, 2] = skip_token_id

        # last 2 states
        arcs[-2, :3] = torch.tensor((num_grid_states - 1, num_grid_states, blank_id), dtype=torch.int32, device=device)
        arcs[-1, :3] = torch.tensor((num_grid_states, num_grid_states + 1, -1), dtype=torch.int32, device=device)

        # sequence indices, time indices
        olabels = torch.div(arcs[:, 0], (text_length + 1), rounding_mode="floor")  # arcs[:, 0] // (text_length + 1)
        unit_positions = arcs[:, 0] % (text_length + 1)
        # last state: final
        olabels[-1] = -1
        unit_positions[-1] = -1

        # relabel states to speedup k2 computations, reusing method from original GraphRnntLoss
        arcs[:-2, 0] = self.relabel_states(arcs[:-2, 0], text_length + 1, num_frames)
        arcs[:-3, 1] = self.relabel_states(arcs[:-3, 1], text_length + 1, num_frames)

        # sort by start state - required in k2
        indices = torch.argsort(arcs[:, 0], dim=0)
        sorted_arcs = arcs[indices]
        olabels = olabels[indices]
        unit_positions = unit_positions[indices]

        rnnt_graph = k2.Fsa(sorted_arcs, olabels)
        rnnt_graph.unit_positions = unit_positions
        return rnnt_graph

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
        device = logits.device

        # construct computational lattices (composed or directly grids)
        # the method is reused from GraphRnntLoss, and retrieves composition or lattice directly
        # for each item in the batch, based on self.use_grid_implementation parameter
        target_fsas_vec = self.get_graphs_batched(logits_lengths, targets, target_lengths, vocab_size)

        skip_frame_id = vocab_size + self.skip_frame_id_rel
        skip_token_id = vocab_size + self.skip_token_id_rel

        cast_context = force_float32_context() if self.cast_to_float32 else nullcontext()
        with cast_context:
            log_probs = F.log_softmax(logits, dim=-1)
            with torch.no_grad():
                last_transition_mask = target_fsas_vec.labels == -1
                skip_token_transition_mask = target_fsas_vec.labels == skip_token_id
                skip_frame_transition_mask = target_fsas_vec.labels == skip_frame_id

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

            # NB: do not assign scores -> modify, k2 will not update all scores correctly (modify -> assign)
            scores = log_probs[batch_indices, time_indices, unit_indices, text_units]
            # fix weights for the arcs to the last state
            scores[last_transition_mask] = 0

            scores[skip_frame_transition_mask] = self.skip_frame_penalty

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
                    max_scores = max_logprob[batch_indices, time_indices, unit_indices]
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
                    sum_scores = sum_logprobs[batch_indices, time_indices, unit_indices]
                    scores = torch.where(skip_token_transition_mask, sum_scores, scores)
                    scores[skip_token_transition_mask] += self.skip_token_penalty
                case "meanexcl":
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
                    mean_logprobs = torch.logsumexp(
                        log_probs_modified[..., : self.blank], dim=-1, keepdim=False
                    ) - math.log(vocab_size - 2)
                    mean_scores = mean_logprobs[batch_indices, time_indices, unit_indices]
                    scores = torch.where(skip_token_transition_mask, mean_scores, scores)
                    scores[skip_token_transition_mask] += self.skip_token_penalty
                case _:
                    raise NotImplementedError

            target_fsas_vec.scores = scores
            if self.use_alignment_prob > 0.0 and random.random() < self.use_alignment_prob:
                shortest_paths = k2.shortest_path(target_fsas_vec, use_double_scores=True)
                shortest_paths_batch_indices = torch.repeat_interleave(
                    torch.arange(batch_size, device=device, dtype=torch.int64),
                    torch.tensor(
                        [shortest_paths.arcs.index(0, i)[0].values().shape[0] for i in range(batch_size)],
                        device=device,
                    ),
                )
                aligned_to_special_tokens = torch.logical_or(
                    shortest_paths.labels == skip_frame_id, shortest_paths.labels == skip_token_id
                )
                aligned_batch_indices = (
                    torch.unique((shortest_paths_batch_indices + 1) * aligned_to_special_tokens) - 1
                )
                aligned_batch_indices_set = set(aligned_batch_indices.tolist())
                not_aligned_batch_indices_set = set(range(batch_size)) - aligned_batch_indices_set
                not_aligned_batch_indices = torch.tensor(list(not_aligned_batch_indices_set), device=device)
                batch_mask = (batch_indices.unsqueeze(0) == not_aligned_batch_indices.unsqueeze(-1)).any(dim=0)
                skip_transition_mask = torch.logical_or(skip_token_transition_mask, skip_frame_transition_mask)
                scores[torch.logical_and(skip_transition_mask, batch_mask)] = float("-inf")
                target_fsas_vec.scores = scores
            scores = -1 * target_fsas_vec.get_tot_scores(use_double_scores=self.double_scores, log_semiring=True)
            return scores
