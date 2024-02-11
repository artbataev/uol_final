import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from min_rnnt.models import MinRNNTModel


def main():
    cfg = OmegaConf.load("conf/fast_conformer_transducer_min.yaml")
    OmegaConf.resolve(cfg)

    cfg.trainer.devices = "auto"
    cfg.trainer.accelerator = "cpu"
    cfg.trainer.strategy = "auto"
    cfg.trainer.max_epochs = 10
    cfg.trainer.val_check_interval = 1.0
    cfg.trainer.log_every_n_steps = 100
    trainer = pl.Trainer(**cfg.trainer)

    cfg.model.train_ds.manifest_filepath = "/Users/vbataev/code/asr_particles/manifests/librispeech/dev_clean.json"
    cfg.model.train_ds.batch_size = 8
    cfg.model.train_ds.num_workers = 0
    cfg.model.validation_ds.manifest_filepath = (
        "/Users/vbataev/code/asr_particles/manifests/librispeech/dev_clean.json"
    )
    cfg.model.validation_ds.batch_size = 8
    cfg.model.validation_ds.num_workers = 0
    cfg.model.tokenizer.dir = (
        "/Users/vbataev/code/asr_particles/tokenizer_models/librispeech/tokenizer_stt_en_conformer_transducer_large_ls"
    )

    model = MinRNNTModel(cfg.model, trainer=trainer)
    trainer.validate(model)
    # trainer.fit(model)


if __name__ == "__main__":
    main()
