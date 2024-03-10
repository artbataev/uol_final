# This is the part of the Final Project
# "RNN-Transducer-based Losses for Speech Recognition on Noisy Targets"
# Originally published in https://github.com/artbataev/uol_final

from contextlib import nullcontext

import k2
import torch
import torch.nn.functional as F
from nemo.collections.asr.parts.k2.graph_transducer import GraphRnntLoss, force_float32_context


class GraphStarTransducerLoss(GraphRnntLoss):
    """
    Original implementation of Star-Transducer loss.
    The implementation is based on "Graph-based framework  for RNN-Transducer losses",
    For the original implementation see
    https://github.com/NVIDIA/NeMo/blob/v1.21.0/nemo/collections/asr/parts/k2/graph_transducer.py

    In this loss we augment the RNN-Transducer with additional skip_frame arcs parallel to blank arcs
    to allow the loss to take into account alignments, where some time frames are skipped.
    The loss is useful for training RNN-T system with partial transcripts (deletions in ground truth texts).
    """

    def __init__(
        self,
        blank: int,
        skip_frame_penalty: float = 0.0,
        return_graph: bool = False,
        use_grid_implementation=True,
        connect_composed=False,
        double_scores=False,
        cast_to_float32=False,
    ):
        """
        Init method

        :param blank: blank label index
        :param skip_frame_penalty: weight of epsilon transitions, 0 means no penalty (default)
        :param return_graph: return graph with loss from forward (False by default)
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
        self.return_graph = return_graph
        self.skip_frame_penalty = skip_frame_penalty

    def get_unit_schema(self, units_tensor: torch.Tensor, vocab_size: int) -> "k2.Fsa":
        """
        Construct WFST from text units.
        Compared to standard RNN-T we add <skip frame> self-loops along with blank self-loops.
        """
        blank_id = self.blank
        skip_frame_id = vocab_size
        device = units_tensor.device
        text_len = units_tensor.shape[0]

        # units_transitions: scr, dest, label, score (k2 convention)
        units_transitions = torch.zeros(((text_len + 1) * 3, 4), dtype=torch.int32, device=device)
        text_indices = torch.arange(0, text_len + 1, dtype=torch.int32, device=device)
        # fill blank labels: self-loops, each 3rd element
        units_transitions[0:-1:3, 0] = text_indices  # from state
        units_transitions[0:-1:3, 1] = text_indices  # to state
        units_transitions[0:-1:3, 2] = blank_id

        # skip_frame labels: each 3rd element starting from 1, self-loops
        units_transitions[1:-1:3, 0] = text_indices  # from state
        units_transitions[1:-1:3, 1] = text_indices  # to state
        units_transitions[1:-1:3, 2] = skip_frame_id

        # text labels
        units_transitions[2::3, 0] = text_indices  # from state
        units_transitions[2::3, 1] = text_indices + 1  # to state -> next to text_indices
        units_transitions[2:-1:3, 2] = units_tensor  # labels: text units

        # last transition to final state, ilabel=-1 (special for k2)
        units_transitions[-1, 2] = -1
        # output labels: same as input labels = text units
        olabels = units_transitions[:, 2].detach().clone()

        # constract fsa
        fsa_units_start = k2.Fsa(units_transitions, olabels)
        # fill in unit positions
        fsa_units_start.unit_positions = text_indices.expand(3, -1).transpose(0, 1).flatten()
        fsa_units_start.unit_positions[-1] = -1  # last transition, k2 convention
        return fsa_units_start

    def get_temporal_schema(self, num_frames: int, vocab_size: int, device: torch.device) -> "k2.Fsa":
        """
        Construct WFST for temporal schema.
        Parallel to blank transitions we add 'skip frame' transitions
        """
        # input labels: all text units
        # output labels: time indices
        blank_id = self.blank

        temporal_transitions = torch.zeros((num_frames * (vocab_size + 1), 4), dtype=torch.int32, device=device)
        time_ids = torch.arange(0, num_frames, dtype=torch.int32, device=device)
        start_states = time_ids.expand(vocab_size + 1, num_frames).transpose(0, 1).flatten()
        # self-loops: emit all tokens from the vocabulary;
        # then replace transitions to next state when necessary
        temporal_transitions[:, 0] = start_states  # from
        temporal_transitions[:, 1] = start_states  # to
        temporal_transitions[:, 2] = (
            torch.arange(0, vocab_size + 1, dtype=torch.int32, device=device)
            .expand(num_frames, vocab_size + 1)
            .flatten()
        )
        # add blank transitions: forward (by time)
        temporal_transitions[blank_id : -1 : vocab_size + 1, 1] = time_ids + 1
        # skip_frame arcs: forward (parallel to blank)
        temporal_transitions[vocab_size : -1 : vocab_size + 1, 1] = (time_ids + 1)[:-1]

        # add transition to the last final state (should be separated in k2)
        temporal_transitions[-1, :3] = torch.tensor((num_frames, num_frames + 1, -1), dtype=torch.int32, device=device)

        # output symbols: position in the sequence, same as start states for all arcs
        olabels = temporal_transitions[:, 0].detach().clone()
        # special for k2 transition to the last state with label -1
        olabels[-1] = -1

        # construct FSA
        fsa_temporal_start = k2.Fsa(temporal_transitions, olabels)
        # sort arcs - necessary for composition
        fsa_temporal_start = k2.arc_sort(fsa_temporal_start)
        return fsa_temporal_start

    def get_grid(self, units_tensor: torch.Tensor, num_frames: int, vocab_size: int) -> "k2.Fsa":
        """
        Directly construct lattice for Star-Transducer
        We use grid labeled enumerated by rows/columns
        """
        blank_id = self.blank
        # skip frame id - outside vocab
        skip_frame_id = vocab_size

        text_length = units_tensor.shape[0]
        device = units_tensor.device

        # construct grid: num-frames * (text-length + 1);
        # we add +1 to test length, since the text is padded with SOS,
        # and blank should be emitted at the end
        num_grid_states = num_frames * (text_length + 1)
        num_blank_arcs = (num_frames - 1) * (text_length + 1)
        num_text_arcs = text_length * num_frames
        transitions = torch.zeros((num_blank_arcs * 2 + num_text_arcs + 2, 4), dtype=torch.int32, device=device)
        # blank and skip frame transitions - parallel
        from_states = torch.arange(num_blank_arcs, device=device)
        to_states = from_states + (text_length + 1)  # by grid construction

        # blank transitions
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
        to_states = from_states + 1  # states are enumerated by text transitions
        text_labels = units_tensor.expand(num_frames, -1).flatten()
        transitions[num_blank_arcs * 2 : num_blank_arcs * 2 + num_text_arcs, 0] = from_states
        transitions[num_blank_arcs * 2 : num_blank_arcs * 2 + num_text_arcs, 1] = to_states
        transitions[num_blank_arcs * 2 : num_blank_arcs * 2 + num_text_arcs, 2] = text_labels

        # last 2 states
        # emit blank last
        transitions[-2, :3] = torch.tensor(
            (num_grid_states - 1, num_grid_states, blank_id), dtype=torch.int32, device=device
        )
        # special to k2 transition to final state with -1 label
        transitions[-1, :3] = torch.tensor(
            (num_grid_states, num_grid_states + 1, -1), dtype=torch.int32, device=device
        )

        # unit indices, time indices
        # construct time indices
        # transitions[:, 0] // (text_length + 1)
        olabels = torch.div(transitions[:, 0], (text_length + 1), rounding_mode="floor")
        # construct unit indices
        unit_positions = transitions[:, 0] % (text_length + 1)
        # last state: final - special `-1` transition
        olabels[-1] = -1
        unit_positions[-1] = -1

        # relabel states to speedup k2 computations, reusing method from original GraphRnntLoss
        transitions[:-2, 0] = self.relabel_states(transitions[:-2, 0], text_length + 1, num_frames)
        transitions[:-3, 1] = self.relabel_states(transitions[:-3, 1], text_length + 1, num_frames)

        # sort by start state - required in k2
        indices = torch.argsort(transitions[:, 0], dim=0)
        # store sorted components: transitions, output labels (time indices), unit positions
        sorted_arcs = transitions[indices]
        olabels = olabels[indices]
        unit_positions = unit_positions[indices]

        start_graph = k2.Fsa(sorted_arcs, olabels)
        start_graph.unit_positions = unit_positions
        return start_graph

    def forward(
        self,
        acts: torch.Tensor,
        labels: torch.Tensor,
        act_lens: torch.Tensor,
        label_lens: torch.Tensor,
    ):
        """
        Forward method is similar to RNN-T Graph-Transducer forward method,
        see https://github.com/NVIDIA/NeMo/blob/v1.21.0/nemo/collections/asr/parts/k2/graph_transducer.py
        We customize skip_frames penalty (hyperparameter of the loss)
        """
        # argument names are consistent with NeMo, see RNNTLoss.forward
        logits, targets, logits_lengths, target_lengths = acts, labels, act_lens, label_lens

        # logits: Batch x Time x TextUnits+1 x Vocab
        vocab_size = logits.shape[-1]
        batch_size = logits.shape[0]
        device = logits.device
        # construct computational lattices (composed or directly grids)
        # the method is reused from GraphRnntLoss, and retrieves composition or lattice directly
        # for each item in the batch, based on self.use_grid_implementation parameter
        target_fsas_vec = self.get_graphs_batched(logits_lengths, targets, target_lengths, vocab_size)

        cast_context = force_float32_context() if self.cast_to_float32 else nullcontext()
        with cast_context:
            # use activation: log softmax (log probabilities - output of Joint)
            log_probs = F.log_softmax(logits, dim=-1)

            with torch.no_grad():
                # mask for last transition - for all graphs
                last_transition_mask = target_fsas_vec.labels == -1
                # mask for skip frame transitions - for all graphs
                skip_frame_transition_mask = target_fsas_vec.labels == vocab_size

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

                # fill in the indices outside the logits with 0, replace later
                # last transitions
                batch_indices.masked_fill_(last_transition_mask, 0)
                time_indices.masked_fill_(last_transition_mask, 0)
                unit_indices.masked_fill_(last_transition_mask, 0)
                text_units.masked_fill_(last_transition_mask, 0)

                # skip frames transitions
                batch_indices.masked_fill_(skip_frame_transition_mask, 0)
                time_indices.masked_fill_(skip_frame_transition_mask, 0)
                unit_indices.masked_fill_(skip_frame_transition_mask, 0)
                text_units.masked_fill_(skip_frame_transition_mask, 0)

            # fill in transition scores
            scores = log_probs[batch_indices, time_indices, unit_indices, text_units]
            # fix weights for the transitions to the last state (special for k2)
            scores[last_transition_mask] = 0
            # assign skip_frame penalty to skip_frame arcs
            scores[skip_frame_transition_mask] = self.skip_frame_penalty

            # assign scores to graphs
            target_fsas_vec.scores = scores

            # compute loss: full-sum loss using k2 graph method
            scores = -1 * target_fsas_vec.get_tot_scores(use_double_scores=self.double_scores, log_semiring=True)
            if self.return_graph:
                return scores, target_fsas_vec
            return scores
