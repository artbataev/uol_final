# This is the part of the Final Project
# "RNN-Transducer-based Losses for Speech Recognition on Noisy Targets"
# Originally published in https://github.com/artbataev/uol_final

import logging
import math
from typing import Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor, ConformerEncoder, SpectrogramAugmentation
from nemo.collections.asr.parts.k2.graph_transducer import GraphRnntLoss
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from omegaconf import DictConfig, open_dict

from min_rnnt.decoding import RNNTDecodingWrapper
from min_rnnt.losses import GraphBypassTransducerLoss, GraphStarTransducerLoss, GraphTargetRobustTransducerLoss
from min_rnnt.metrics import ExtendedWordErrorRate
from min_rnnt.modules import MinJoint, MinPredictionNetwork


class MinRNNTModel(ASRModel, ASRBPEMixin):
    """
    Minimal RNN-T model with custom MinJoint and MinPredictionNetwork modules,
    reuses Encoder and data loader from NeMo,
    customized from
    https://github.com/NVIDIA/NeMo/blob/v1.21.0/nemo/collections/asr/models/rnnt_bpe_models.py
    Main customization:
    - our implementation of Joint and Prediction network
    - custom losses
    - our metric for WER to log not only WER, but separately its components (DEL, INS, SUB)
    Reused:
    - encoder (Conformer)
    - data loaders
    - logging (including automatic WandB logging)
    """

    val_wer: nn.ModuleList

    def __init__(self, cfg: DictConfig, trainer: pl.Trainer = None):
        # setup SentencePiece tokenizer
        self._setup_tokenizer(cfg.tokenizer)

        # get vocabulary size and blank index (outside the vocabulary)
        vocabulary = self.tokenizer.tokenizer.get_vocab()
        vocabulary_size = len(vocabulary)
        if hasattr(self.tokenizer, "pad_id") and self.tokenizer.pad_id > 0:
            self.text_pad_id = self.tokenizer.pad_id
        else:
            self.text_pad_id = vocabulary_size
        self.blank_index = vocabulary_size

        super().__init__(cfg=cfg, trainer=trainer)

        # preprocessor and spec augment modules from NeMo
        self.preprocessor = AudioToMelSpectrogramPreprocessor(**cfg.preprocessor)
        self.spec_aug = SpectrogramAugmentation(**cfg.spec_augment) if cfg.spec_augment else None

        # Encoder part - Conformer Encoder from NeMo
        self.encoder = ConformerEncoder(**cfg.encoder)

        # Prediction and Joint networks - we use our implementation instead of NeMo
        # instantiate prediction network
        with open_dict(self.cfg):
            self.cfg.prediction_network["vocab_size"] = vocabulary_size
        prediction_network = MinPredictionNetwork(**self.cfg.prediction_network)

        # instantiate joint netowkr
        with open_dict(self.cfg):
            self.cfg.joint["output_size"] = vocabulary_size + 1  # vocab + blank
        joint = MinJoint(**self.cfg.joint)

        # decoding wrapper module for Joint and Prediction networks
        self.decoding = RNNTDecodingWrapper(
            prediction_network=prediction_network,
            joint=joint,
            blank_index=self.blank_index,
            max_symbols_per_step=self.cfg.decoding.get("max_symbols", 10),
        )

        # loss parameters
        self.skip_token_decay = self.cfg.loss.get("skip_token_decay", 1.0)
        self.skip_token_penalty_end = self.cfg.loss.get("skip_token_penalty_end", 0.0)
        self.skip_token_decay_min_epoch = 3
        if self.cfg.loss.loss_name == "rnnt":
            # standard RNN-T (Graph RNN-T loss from NeMo)
            self.loss = GraphRnntLoss(blank=self.blank_index, double_scores=True)
        elif self.cfg.loss.loss_name == "star_t":
            # original: Star Transducer loss
            self.loss = GraphStarTransducerLoss(
                blank=self.blank_index,
                skip_frame_penalty=self.cfg.loss.get("skip_frame_penalty", 0.0),
                double_scores=True,
            )
        elif self.cfg.loss.loss_name == "bypass_t":
            # original: Bypass Transducer loss
            self.loss = GraphBypassTransducerLoss(
                blank=self.blank_index,
                skip_token_penalty=self.cfg.loss.get("skip_token_penalty", 0.0),
                skip_token_mode=self.cfg.loss.get("skip_token_mode", "sumexcl"),
                double_scores=True,
            )
        elif self.cfg.loss.loss_name == "trt":
            # original: Target Robust Transducer loss
            # NB: use_alignment_prob is a parameter for future exploration, not used in the project (0 by default)
            self.loss = GraphTargetRobustTransducerLoss(
                blank=self.blank_index,
                skip_frame_penalty=self.cfg.loss.get("skip_frame_penalty", 0.0),
                skip_token_penalty=self.cfg.loss.get("skip_token_penalty", 0.0),
                skip_token_mode=self.cfg.loss.get("skip_token_mode", "sumexcl"),
                use_alignment_prob=self.cfg.loss.get("use_alignment_prob", 0.0),
                double_scores=True,
            )
        else:
            raise NotImplementedError(f"loss {self.cfg.loss.loss_name} not supported")

        # WER metric to compute WER in training
        self.wer = ExtendedWordErrorRate(dist_sync_on_step=True)

    def forward(self, audio: torch.Tensor, audio_lengths: torch.Tensor):
        """Forward step: encoder only, following NeMo model style"""
        audio_features, audio_features_lengths = self.preprocessor(
            input_signal=audio,
            length=audio_lengths,
        )
        # spec augnment - apply only in training
        if self.spec_aug is not None and self.training:
            audio_features = self.spec_aug(input_spec=audio_features, length=audio_features_lengths)
        # get encoder output
        encoded_audio, encoded_audio_lengths = self.encoder(audio_signal=audio_features, length=audio_features_lengths)
        return encoded_audio, encoded_audio_lengths

    def training_step(self, batch, batch_nb):
        """Training step: compute and return loss"""
        audio, audio_lengths, targets, targets_lengths = batch
        # get encoder ouptut
        encoded_audio, encoded_audio_lengths = self.forward(audio=audio, audio_lengths=audio_lengths)
        # get joint output (forward pass, training)
        joint = self.decoding(
            encoder_output=encoded_audio,
            encoder_lengths=encoded_audio_lengths,
            targets=targets,
            target_lengths=targets_lengths,
        )
        # compute loss value
        loss_value = self.loss(acts=joint, act_lens=encoded_audio_lengths, labels=targets, label_lens=targets_lengths)
        # mean volume reduction according to Fast Conformer original recipe in NeMo
        loss_value = loss_value.sum() / targets_lengths.sum()

        assert self.trainer is not None, "Trainer should be set if training_step is called"
        # log the training state
        detailed_logs = dict()
        detailed_logs["train_loss"] = loss_value.item()
        detailed_logs["learning_rate"] = self._optimizer.param_groups[0]["lr"]
        detailed_logs["global_step"] = self.trainer.global_step
        sample_id = self.trainer.global_step

        # for monitoring training: compute WER and components on training data
        if sample_id % self.trainer.log_every_n_steps == 0:
            # decode and log in training
            with torch.no_grad():
                # set to eval mode
                self.eval()
                # decode the encoder output
                hyps = self.decoding.greedy_decode(encoder_output=encoded_audio, encoder_lengths=encoded_audio_lengths)
                # convert hypotheses to strings
                hyps_str = [self.tokenizer.ids_to_text(current_hyp) for current_hyp in hyps]
                batch_size = targets.shape[0]
                # convert reference to strings
                refs_str = [
                    self.tokenizer.ids_to_text(targets[i, : targets_lengths[i]].tolist()) for i in range(batch_size)
                ]
                # compute wer and components
                self.wer.update(preds=hyps_str, target=refs_str)
                wer_value = self.wer.compute()
                detailed_logs["training_wer"] = wer_value["wer"]
                detailed_logs["training_sub"] = wer_value["substitutions"]
                detailed_logs["training_del"] = wer_value["deletions"]
                detailed_logs["training_ins"] = wer_value["insertions"]
                self.wer.reset()
                # set to train mode
                self.train()
        # log results
        self.log_dict(detailed_logs)
        return {"loss": loss_value}

    def on_train_epoch_start(self) -> None:
        """Apply token decay for the penalty"""
        if self.cfg.loss.loss_name in {"bypass_t", "trt"}:
            initial_skip_token_penalty = self.cfg.loss.get("skip_token_penalty", 0.0)
            if self.trainer.current_epoch > self.skip_token_decay_min_epoch:
                # compute penalty = initial_penalty * (decay ^ (epoch - start_decay_epoch))
                self.loss.skip_token_penalty = initial_skip_token_penalty * (
                    self.skip_token_decay ** (self.trainer.current_epoch - self.skip_token_decay_min_epoch)
                )
                # respect maximum token penalty
                if self.loss.skip_token_penalty > self.skip_token_penalty_end:
                    self.loss.skip_token_penalty = self.skip_token_penalty_end
            # log penalty
            self.log("skip_token_penalty", self.loss.skip_token_penalty)

    def validation_step(self, batch, batch_nb, dataloader_idx=0):
        """Validation step: decode and compute WER"""
        audio, audio_lengths, targets, targets_lengths = batch
        # get encoder output
        encoded_audio, encoded_audio_lengths = self.forward(audio=audio, audio_lengths=audio_lengths)
        # greedy decode hypotheses
        hyps = self.decoding.greedy_decode(encoder_output=encoded_audio, encoder_lengths=encoded_audio_lengths)
        # convert hypotheses to strings
        hyps_str = [self.tokenizer.ids_to_text(current_hyp) for current_hyp in hyps]
        # get references as strings
        batch_size = targets.shape[0]
        refs_str = [self.tokenizer.ids_to_text(targets[i, : targets_lengths[i]].tolist()) for i in range(batch_size)]
        # log first hypothesis and reference
        logging.info(f"val ref: {refs_str[0]}")
        logging.info(f"val hyp: {hyps_str[0]}")
        # compute wer and components
        self.val_wer[dataloader_idx].update(preds=hyps_str, target=refs_str)

    def on_validation_start(self):
        # reset validation metric when the validation starts
        for submodule in self.val_wer:
            submodule.reset()

    def multi_validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]], dataloader_idx: int = 0):
        """
        Aggregate wer metric for all validation dataloader
        (first dataloader is used for checkpoint selection, other - only for logging)
        """
        wer_detailed = self.val_wer[dataloader_idx].compute()
        if wer_detailed["wer"].isnan():
            # when resuming the function can be called without real validation for all datasets,
            # thus we do not return the metric
            return {"log": {}}
        return {
            "log": {
                "val_wer": wer_detailed["wer"],
                "val_del": wer_detailed["deletions"],
                "val_ins": wer_detailed["insertions"],
                "val_sub": wer_detailed["substitutions"],
            }
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Return hypotheses predicted from the audio"""
        audio, audio_lengths, targets, targets_lengths = batch
        encoded_audio, encoded_audio_lengths = self.forward(audio=audio, audio_lengths=audio_lengths)
        hyps = self.decoding.greedy_decode(encoder_output=encoded_audio, encoder_lengths=encoded_audio_lengths)
        return hyps

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        """Training data setup: derived from NeMo to reuse dataloaders"""
        train_data_config["shuffle"] = True
        self._update_dataset_config(dataset_name="train", config=train_data_config)
        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)
        # this is a fix for tqdm bar ported from NeMo
        if (
            self._train_dl is not None
            and hasattr(self._train_dl, "dataset")
            and isinstance(self._train_dl.dataset, torch.utils.data.IterableDataset)
        ):
            if self.trainer is not None and isinstance(self.trainer.limit_train_batches, float):
                self.trainer.limit_train_batches = int(
                    self.trainer.limit_train_batches
                    * math.ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config["batch_size"])
                )

    def setup_multiple_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        """Setup multiple validation data, similar to NeMo models"""
        super().setup_multiple_validation_data(val_data_config=val_data_config)
        num_val_loaders = len(self._validation_dl) if isinstance(self._validation_dl, list) else 1
        # init val_wer metrics
        self.val_wer = nn.ModuleList([ExtendedWordErrorRate(dist_sync_on_step=True) for _ in range(num_val_loaders)])

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        """Setup validation data, similar to NeMo models"""
        # avoid shuffling manifest - fixed order of utterances
        val_data_config["shuffle"] = False
        # store dataset config
        self._update_dataset_config(dataset_name="validation", config=val_data_config)
        # get dataloader
        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        """Setup test data, similar to NeMo models"""
        # avoid shuffling manifest - fixed order of utterances
        test_data_config["shuffle"] = False
        # store config
        self._update_dataset_config(dataset_name="test", config=test_data_config)
        # get dataloader
        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        """Taken from NeMo ASR Model as described above to reuse dataloaders"""
        # get dataset
        dataset = audio_to_text_dataset.get_audio_to_text_bpe_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            tokenizer=self.tokenizer,
            preprocessor_cfg=self.cfg.get("preprocessor", None),
        )

        if dataset is None:
            return None

        # fix for shuffling: if dataset is iterable - no shuffling
        shuffle = config["shuffle"]
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

        # use collate fn from the dataset or the dataset which is a combination of other datasets
        if hasattr(dataset, "collate_fn"):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], "collate_fn"):
            # dataset -> list of datasets
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # dataset -> list of datasets -> list of datasets
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        # instantiate and return dataloader
        batch_size = config["batch_size"]
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=False,
            shuffle=shuffle,
            num_workers=config.get("num_workers", 0),
            pin_memory=config.get("pin_memory", True),
        )

    def transcribe(self, paths2audio_files: List[str], batch_size: int = 4, verbose: bool = True) -> List[str]:
        """
        We avoid implementing transcribe function for NeMo model,
        since for our project we need only validation/testing logic.
        We also have a predict_step that returns hypotheses
        """
        raise NotImplementedError
