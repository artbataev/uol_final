import logging
import math
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor, ConformerEncoder, SpectrogramAugmentation
from nemo.collections.asr.parts.k2.graph_transducer import GraphRnntLoss
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from omegaconf import DictConfig, open_dict
from torchmetrics.text import WordErrorRate

from min_rnnt.decoding import RNNTDecodingWrapper
from min_rnnt.modules import MinJoint, MinPredictionNetwork


class MinRNNTModel(ASRModel, ASRBPEMixin):
    def __init__(self, cfg: DictConfig, trainer: pl.Trainer = None):
        self._setup_tokenizer(cfg.tokenizer)
        vocabulary = self.tokenizer.tokenizer.get_vocab()
        vocabulary_size = len(vocabulary)
        if hasattr(self.tokenizer, "pad_id") and self.tokenizer.pad_id > 0:
            self.text_pad_id = self.tokenizer.pad_id
        else:
            self.text_pad_id = vocabulary_size
        self.blank_index = vocabulary_size

        super().__init__(cfg=cfg, trainer=trainer)
        self.preprocessor = AudioToMelSpectrogramPreprocessor(**cfg.preprocessor)
        self.spec_aug = SpectrogramAugmentation(**cfg.spec_augment) if cfg.spec_augment else None
        self.encoder = ConformerEncoder(**cfg.encoder)

        with open_dict(self.cfg):
            self.cfg.prediction_network["vocab_size"] = vocabulary_size
        prediction_network = MinPredictionNetwork(**self.cfg.prediction_network)
        with open_dict(self.cfg):
            self.cfg.joint["output_size"] = vocabulary_size + 1  # vocab + blank
        joint = MinJoint(**self.cfg.joint)
        self.decoding = RNNTDecodingWrapper(
            prediction_network=prediction_network,
            joint=joint,
            blank_index=self.blank_index,
            max_symbols_per_step=self.cfg.decoding.get("max_symbols", 10),
        )
        self.loss = GraphRnntLoss(blank=self.blank_index, double_scores=True)
        self.wer = WordErrorRate()
        self.val_wer: List[WordErrorRate] = []

    def forward(self, audio: torch.Tensor, audio_lengths: torch.Tensor):
        # logging.warning(f"audio: {audio.shape}, expected BxT")
        audio_features, audio_features_lengths = self.preprocessor(
            input_signal=audio,
            length=audio_lengths,
        )
        # logging.warning(f"audio_features: {audio_features.shape}, expected BxDxT")
        if self.spec_aug is not None and self.training:
            audio_features = self.spec_aug(input_spec=audio_features, length=audio_features_lengths)
        encoded_audio, encoded_audio_lengths = self.encoder(audio_signal=audio_features, length=audio_features_lengths)
        # logging.warning(f"encoded audio {encoded_audio.shape}, expected BxDxT")
        return encoded_audio, encoded_audio_lengths

    def training_step(self, batch, batch_nb):
        audio, audio_lengths, targets, targets_lengths = batch
        encoded_audio, encoded_audio_lengths = self.forward(audio=audio, audio_lengths=audio_lengths)
        joint = self.decoding(
            encoder_output=encoded_audio,
            encoder_lengths=encoded_audio_lengths,
            targets=targets,
            target_lengths=targets_lengths,
        )
        loss_value = self.loss(acts=joint, act_lens=encoded_audio_lengths, labels=targets, label_lens=targets_lengths)
        loss_value = loss_value.sum() / targets_lengths.sum()  # TODO: make parameter - reduction

        assert self.trainer is not None, "Trainer should be set if training_step is called"
        detailed_logs = dict()
        detailed_logs["train_loss"] = loss_value.item()
        detailed_logs["learning_rate"] = self._optimizer.param_groups[0]["lr"]
        detailed_logs["global_step"] = self.trainer.global_step
        sample_id = self.trainer.global_step
        if sample_id % self.trainer.log_every_n_steps == 0:
            with torch.no_grad():
                self.eval()
                hyps = self.decoding.greedy_decode(encoder_output=encoded_audio, encoder_lengths=encoded_audio_lengths)
                hyps_str = [self.tokenizer.ids_to_text(current_hyp) for current_hyp in hyps]
                batch_size = targets.shape[0]
                refs_str = [
                    self.tokenizer.ids_to_text(targets[i, : targets_lengths[i]].tolist()) for i in range(batch_size)
                ]
                self.wer.update(preds=hyps_str, target=refs_str)
                wer_value = self.wer.compute()
                detailed_logs["training_wer"] = wer_value
                self.wer.reset()
                self.train()
        self.log_dict(detailed_logs)
        return {"loss": loss_value}

    def validation_step(self, batch, batch_nb, dataloader_idx=0):
        audio, audio_lengths, targets, targets_lengths = batch
        encoded_audio, encoded_audio_lengths = self.forward(audio=audio, audio_lengths=audio_lengths)
        hyps = self.decoding.greedy_decode(encoder_output=encoded_audio, encoder_lengths=encoded_audio_lengths)
        hyps_str = [self.tokenizer.ids_to_text(current_hyp) for current_hyp in hyps]
        batch_size = targets.shape[0]
        refs_str = [self.tokenizer.ids_to_text(targets[i, : targets_lengths[i]].tolist()) for i in range(batch_size)]
        logging.info(f"val ref: {refs_str[0]}")
        logging.info(f"val hyp: {hyps_str[0]}")
        self.val_wer[dataloader_idx].update(preds=hyps_str, target=refs_str)

    def on_validation_start(self):
        num_val_loaders = len(self._validation_dl) if isinstance(self._validation_dl, list) else 1
        self.val_wer = [WordErrorRate() for _ in range(num_val_loaders)]

    def multi_validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]], dataloader_idx: int = 0):
        return {"log": {"val_wer": self.val_wer[dataloader_idx].compute()}}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError  # TODO

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        train_data_config["shuffle"] = True
        self._update_dataset_config(dataset_name="train", config=train_data_config)
        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)
        # TODO: fix tqdm bar?
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

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        val_data_config["shuffle"] = False
        self._update_dataset_config(dataset_name="validation", config=val_data_config)
        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        pass

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
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

        shuffle = config["shuffle"]
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

        if hasattr(dataset, "collate_fn"):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], "collate_fn"):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config["batch_size"],
            collate_fn=collate_fn,
            drop_last=config.get("drop_last", False),
            shuffle=shuffle,
            num_workers=config.get("num_workers", 0),
            pin_memory=config.get("pin_memory", False),
        )

    def transcribe(self, paths2audio_files: List[str], batch_size: int = 4, verbose: bool = True) -> List[str]:
        raise NotImplementedError
