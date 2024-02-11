from contextlib import nullcontext

import k2
import torch
import torch.nn.functional as F
from nemo.collections.asr.parts.k2.graph_transducer import GraphRnntLoss, force_float32_context


class GraphStarTransducerLoss(GraphRnntLoss):
    """
    This loss is a modified version of Graph-Transducer, see
    https://github.com/NVIDIA/NeMo/blob/v1.21.0/nemo/collections/asr/parts/k2/graph_transducer.py
    We add skip_token arcs parallel to blank arcs to allow the loss to take into account
    alignments, where some time frames are skipped: useful for training RNN-T with partial transcripts
    (deletions in ground truth texts)
    """

    def __init__(
        self,
        blank: int,
        skip_frame_penalty: float = 0.0,
        return_graph: bool = False,
        use_grid_implementation=False,  # TODO: grid impl
        connect_composed=False,
        double_scores=False,
        cast_to_float32=False,
    ):
        """
        Init method

        Args:
            blank: blank label index
            skip_frame_penalty: weight of skip frame transitions, 0 means no penalty (default)
            use_grid_implementation: Whether to use the grid implementation (Grid-Transducer).
            connect_composed: Connect graph after composing unit and temporal schemas
                (only for Compose-Transducer). `connect` operation is slow, it is useful for visualization,
                but not necessary for loss computation.
            double_scores: Use calculation of loss in double precision (float64) in the lattice.
                Does not significantly affect memory usage since the lattice is ~V/2 times smaller than the joint tensor.
            cast_to_float32: Force cast joint tensor to float32 before log-softmax calculation.
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
        """construct WFST from text units"""
        blank_id = self.blank
        skip_frame_id = vocab_size
        device = units_tensor.device
        text_len = units_tensor.shape[0]

        # arcs: scr, dest, label, score
        units_arcs = torch.zeros(((text_len + 1) * 3, 4), dtype=torch.int32, device=device)
        text_indices = torch.arange(0, text_len + 1, dtype=torch.int32, device=device)
        # fill blank labels: self-loops, each 3rd element
        units_arcs[0:-1:3, 0] = text_indices  # from state
        units_arcs[0:-1:3, 1] = text_indices  # to state
        units_arcs[0:-1:3, 2] = blank_id

        # skip_frame labels: each 3rd element starting from 1, self-loops
        units_arcs[1:-1:3, 0] = text_indices  # from state
        units_arcs[1:-1:3, 1] = text_indices  # to state
        units_arcs[1:-1:3, 2] = skip_frame_id

        # text labels
        units_arcs[2::3, 0] = text_indices  # from state
        units_arcs[2::3, 1] = text_indices + 1  # to state
        units_arcs[2:-1:3, 2] = units_tensor  # labels: text
        units_arcs[-1, 2] = -1  # last transition to final state, ilabel=-1 (special for k2)
        olabels = units_arcs[:, 2].detach().clone()  # same as ilabels, text units

        fsa_units = k2.Fsa(units_arcs, olabels)
        fsa_units.unit_positions = text_indices.expand(3, -1).transpose(0, 1).flatten()
        fsa_units.unit_positions[-1] = -1
        return fsa_units

    def get_temporal_schema(self, num_frames: int, vocab_size: int, device: torch.device) -> "k2.Fsa":
        """Construct WFST for temporal schema"""
        # input labels: all text units
        # output labels: time indices
        blank_id = self.blank

        temporal_arcs = torch.zeros((num_frames * (vocab_size + 1), 4), dtype=torch.int32, device=device)
        time_ids = torch.arange(0, num_frames, dtype=torch.int32, device=device)
        # for every state - vocab_size arcs, [0, 1, ..., vocab_size-1, 0, 1, ..., vocab_size-1, ...]
        start_states = time_ids.expand(vocab_size + 1, num_frames).transpose(0, 1).flatten()
        # first: make all arcs - self-loops
        temporal_arcs[:, 0] = start_states  # from
        temporal_arcs[:, 1] = start_states  # to
        temporal_arcs[:, 2] = (
            torch.arange(0, vocab_size + 1, dtype=torch.int32, device=device)
            .expand(num_frames, vocab_size + 1)
            .flatten()
        )

        # blank-arcs: forward
        temporal_arcs[blank_id : -1 : vocab_size + 1, 1] = time_ids + 1
        # skip_frame arcs: forward (parallel to blank)
        temporal_arcs[vocab_size : -1 : vocab_size + 1, 1] = (time_ids + 1)[:-1]

        # transition to the last final state
        temporal_arcs[-1, :3] = torch.tensor((num_frames, num_frames + 1, -1), dtype=torch.int32, device=device)

        # output symbols: position in the sequence, same as start states for all arcs
        olabels = temporal_arcs[:, 0].detach().clone()
        olabels[-1] = -1  # special: last arc to the final state

        fsa_temporal = k2.Fsa(temporal_arcs, olabels)
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
        see https://github.com/NVIDIA/NeMo/blob/v1.21.0/nemo/collections/asr/parts/k2/graph_transducer.py
        We customize skip_frames penalty (hyperparameter of the loss)
        """
        # argument names are consistent with NeMo, see RNNTLoss.forward
        logits, targets, logits_lengths, target_lengths = acts, labels, act_lens, label_lens

        # logits: B x Time x Text+1 x C
        vocab_size = logits.shape[-1]
        batch_size = logits.shape[0]
        device = logits.device
        target_fsas_vec = self.get_graphs_batched(logits_lengths, targets, target_lengths, vocab_size)

        cast_context = force_float32_context() if self.cast_to_float32 else nullcontext()
        with cast_context:
            # activation: log softmax
            log_probs = F.log_softmax(logits, dim=-1)

            with torch.no_grad():
                last_transition_mask = target_fsas_vec.labels == -1
                skip_frame_transition_mask = target_fsas_vec.labels == vocab_size

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

                # eps transitions
                batch_indices.masked_fill_(last_transition_mask, 0)
                time_indices.masked_fill_(last_transition_mask, 0)
                unit_indices.masked_fill_(last_transition_mask, 0)
                text_units.masked_fill_(last_transition_mask, 0)

                # skip frames transitions
                batch_indices.masked_fill_(skip_frame_transition_mask, 0)
                time_indices.masked_fill_(skip_frame_transition_mask, 0)
                unit_indices.masked_fill_(skip_frame_transition_mask, 0)
                text_units.masked_fill_(skip_frame_transition_mask, 0)

            # NB: do not assign scores -> modify, k2 will not update all scores correctly (modify -> assign)
            scores = log_probs[batch_indices, time_indices, unit_indices, text_units]
            # fix weights for the arcs to the last state - special for k2
            scores[last_transition_mask] = 0
            # assign skip_frame penalty to skip_frame arcs
            scores[skip_frame_transition_mask] = self.skip_frame_penalty

            target_fsas_vec.scores = scores

            # compute loss: full-sum
            scores = -1 * target_fsas_vec.get_tot_scores(use_double_scores=self.double_scores, log_semiring=True)
            if self.return_graph:
                return scores, target_fsas_vec
            return scores
