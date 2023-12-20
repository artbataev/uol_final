from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor, ConformerEncoder, SpectrogramAugmentation
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning.utilities.types import STEP_OUTPUT
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

        super().__init__(cfg=cfg, trainer=trainer)
        self.preprocessor = AudioToMelSpectrogramPreprocessor(**cfg.preprocessor)
        self.spec_aug = SpectrogramAugmentation(**cfg.spec_augment) if cfg.spec_augment else None
        self.encoder = ConformerEncoder(**cfg.encoder)
        prediction_network = MinPredictionNetwork(**cfg.prediction_network)
        joint = MinJoint(**cfg.joint)

        self.decoding = RNNTDecodingWrapper(
            prediction_network=prediction_network, joint=joint, blank_index=vocabulary_size, max_symbols_per_step=10
        )
        self.wer = WordErrorRate()

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        pass

    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
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

        shuffle = config['shuffle']
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )
