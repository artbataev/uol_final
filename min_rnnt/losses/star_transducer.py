from contextlib import nullcontext

import k2
import torch
import torch.nn.functional as F
from nemo.collections.asr.parts.k2.graph_transducer import GraphRnntLoss, force_float32_context


class GraphStarTransducerLoss(GraphRnntLoss):
    def __init__(
        self,
        blank: int,
        skip_frame_penalty: float = 0.0,
        use_grid_implementation=False,  # TODO: grid impl
        connect_composed=False,
        double_scores=False,
        cast_to_float32=False,
    ):
        """
        Init method

        Args:
            blank: blank label index
            skip_frame_penalty: weight of epsilon transitions, 0 means no penalty (default)
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
        self.skip_frame_penalty = skip_frame_penalty

    def get_unit_schema(self, units_tensor: torch.Tensor, vocab_size: int) -> "k2.Fsa":
        """
        Get unit schema (target text) graph for Star-Transducer loss (Compose-Transducer).
        Forward arcs represent text labels.

        Example graph: text [1, 2], blank=0. Eps id: 3.

        graph::

                0:0:0                  0:0:1                  0:0:2
              +-------+              +-------+              +-------+
              v       |              v       |              v       |
            +-----------+  1:1:0   +-----------+  2:2:1   +-----------+  -1:-1:-1  #===#
            |     0     | -------> |     1     | -------> |     2     | ---------> H 3 H
            +-----------+          +-----------+          +-----------+            #===#
              ^ 3:3:0 |              ^ 3:3:1 |              ^ 3:3:2 |
              +-------+              +-------+              +-------+

        Args:
            units_tensor: 1d tensor with text units
            vocab_size: number of total labels (vocab size including blank)

        Returns:
            unit schema graph (k2.Fsa).
            Labels: <unit>:<unit>:<unit_position> (k2.Fsa: labels, aux_labels, unit_positions)
        """

        blank_id = self.blank
        eps_id = vocab_size
        device = units_tensor.device
        text_len = units_tensor.shape[0]

        # arcs: scr, dest, label, score
        arcs = torch.zeros(((text_len + 1) * 3, 4), dtype=torch.int32, device=device)
        text_indices = torch.arange(0, text_len + 1, dtype=torch.int32, device=device)
        # blank labels
        arcs[0:-1:3, 0] = text_indices  # from state
        arcs[0:-1:3, 1] = text_indices  # to state
        arcs[0:-1:3, 2] = blank_id

        # eps labels
        arcs[1:-1:3, 0] = text_indices  # from state
        arcs[1:-1:3, 1] = text_indices  # to state
        arcs[1:-1:3, 2] = eps_id

        # text labels
        arcs[2::3, 0] = text_indices  # from state
        arcs[2::3, 1] = text_indices + 1  # to state
        arcs[2:-1:3, 2] = units_tensor  # labels: text
        arcs[-1, 2] = -1  # last transition to final state, ilabel=-1 (special for k2)
        olabels = arcs[:, 2].detach().clone()  # same as ilabels

        fsa_text = k2.Fsa(arcs, olabels)
        fsa_text.unit_positions = text_indices.expand(3, -1).transpose(0, 1).flatten()
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

        fsa_temporal_arcs = torch.zeros((num_frames * (vocab_size + 1), 4), dtype=torch.int32, device=device)
        sequence_states = torch.arange(0, num_frames, dtype=torch.int32, device=device)
        # for every state - vocab_size arcs, [0, 1, ..., vocab_size-1, 0, 1, ..., vocab_size-1, ...]
        start_states = sequence_states.expand(vocab_size + 1, num_frames).transpose(0, 1).flatten()
        # first: make all arcs - self-loops
        fsa_temporal_arcs[:, 0] = start_states  # from
        fsa_temporal_arcs[:, 1] = start_states  # to
        fsa_temporal_arcs[:, 2] = (
            torch.arange(0, vocab_size + 1, dtype=torch.int32, device=device)
            .expand(num_frames, vocab_size + 1)
            .flatten()
        )

        # blank-arcs: forward
        fsa_temporal_arcs[blank_id : -1 : vocab_size + 1, 1] = sequence_states + 1  # blanks
        fsa_temporal_arcs[vocab_size : -1 : vocab_size + 1, 1] = (sequence_states + 1)[:-1]  # [:-1]  # blanks

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
        target_fsas_vec = self.get_graphs_batched(logits_lengths, targets, target_lengths, vocab_size)

        cast_context = force_float32_context() if self.cast_to_float32 else nullcontext()
        with cast_context:
            log_probs = F.log_softmax(logits, dim=-1)
            with torch.no_grad():
                indices = self.get_logits_indices(target_fsas_vec, logits.shape)
                # transition to the last state + eps-transitions
                # use 0 index (for valid index_select) and manually assign score after index_select for this case
                indices[target_fsas_vec.labels == -1] = 0
                indices[target_fsas_vec.labels >= vocab_size] = 0  # eps

            # NB: do not assign scores -> modify, k2 will not update all scores correctly (modify -> assign)
            scores = log_probs.flatten().index_select(-1, indices)
            # fix weights for the arcs to the last state + eps-transitions
            scores[target_fsas_vec.labels == -1] = 0
            scores[target_fsas_vec.labels >= vocab_size] = self.skip_frame_penalty  # eps

            target_fsas_vec.scores = scores
            scores = -1 * target_fsas_vec.get_tot_scores(use_double_scores=self.double_scores, log_semiring=True)
            return scores
