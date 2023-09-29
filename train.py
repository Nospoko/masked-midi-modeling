import os

import hydra
import torch
import wandb
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from omegaconf import OmegaConf
from huggingface_hub import upload_file
from torch.utils.data import Subset, DataLoader
from datasets import load_dataset, concatenate_datasets
from transformers import RobertaForMaskedLM, RobertaConfig

from data.dataset import MidiDataset
from data.tokenizer import QuantizedMidiEncoder

def makedir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def preprocess_dataset(
        dataset_name: list[str], 
        tokenizer: QuantizedMidiEncoder, 
        batch_size: int, num_workers: int, 
        augmentation_probability: float, 
        *, 
        overfit_single_batch: bool = False
    ):
    hf_token = os.environ["HUGGINGFACE_TOKEN"]

    train_ds = []
    val_ds = []
    test_ds = []

    for ds_name in dataset_name:
        tr_ds = load_dataset(ds_name, split="train", use_auth_token=hf_token)
        v_ds = load_dataset(ds_name, split="validation", use_auth_token=hf_token)
        t_ds = load_dataset(ds_name, split="test", use_auth_token=hf_token)

        train_ds.append(tr_ds)
        val_ds.append(v_ds)
        test_ds.append(t_ds)

    train_ds = concatenate_datasets(train_ds)
    val_ds = concatenate_datasets(val_ds)
    test_ds = concatenate_datasets(test_ds)

    train_ds = MidiDataset(train_ds, tokenizer, augmentation_probability=augmentation_probability, masking_probability=0.15)
    val_ds = MidiDataset(val_ds, tokenizer, augmentation_probability=0.0, masking_probability=0.15)
    test_ds = MidiDataset(test_ds, tokenizer, augmentation_probability=0.0, masking_probability=0.15)

    if overfit_single_batch:
        train_ds = Subset(train_ds, indices=range(batch_size))
        val_ds = Subset(val_ds, indices=range(batch_size))
        test_ds = Subset(test_ds, indices=range(batch_size))

    # dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader


def forward_step(
    model: QuantizedMidiEncoder,
    batch: dict[str, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> float:
    input_token_ids = batch["input_token_ids"].to(device)
    tgt_token_ids = batch["tgt_token_ids"].to(device)

    outputs = model(
        input_ids=input_token_ids,
        labels=tgt_token_ids,
    )

    return outputs.loss


@torch.no_grad()
def validation_epoch(
    model: RobertaForMaskedLM,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    # val epoch
    val_loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    loss_epoch = 0.0

    for batch_idx, batch in val_loop:
        # metrics returns loss and additional metrics if specified in step function
        loss = forward_step(model, batch, device)

        val_loop.set_postfix(loss=loss.item())

        loss_epoch += loss.item()

    metrics = {"loss_epoch": loss_epoch / len(dataloader)}
    return metrics


def save_checkpoint(
    model: RobertaForMaskedLM, optimizer: optim.Optimizer, cfg: OmegaConf, save_path: str
):
    # saving models
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        f=save_path,
    )


def upload_to_huggingface(ckpt_save_path: str, cfg: OmegaConf):
    # get huggingface token from environment variables
    token = os.environ["HUGGINGFACE_TOKEN"]

    # upload model to hugging face
    upload_file(ckpt_save_path, path_in_repo=f"{cfg.logger.run_name}.ckpt", repo_id=cfg.paths.hf_repo_id, token=token)


@hydra.main(config_path="configs", config_name="config-default", version_base="1.3.2")
def train(cfg: OmegaConf):
    wandb.login()

    # create dir if they don't exist
    makedir_if_not_exists(cfg.paths.log_dir)
    makedir_if_not_exists(cfg.paths.save_ckpt_dir)

    tokenizer = QuantizedMidiEncoder(dstart_bin=7, duration_bin=7, velocity_bin=7)

    # dataset
    train_dataloader, val_dataloader, _ = preprocess_dataset(
        dataset_name=cfg.train.dataset_name,
        tokenizer=tokenizer,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        augmentation_probability=cfg.train.augmentation_probability,
        overfit_single_batch=cfg.train.overfit_single_batch,
    )

    # validate on quantized maestro
    _, maestro_test, _ = preprocess_dataset(
        dataset_name=["JasiekKaczmarczyk/maestro-sustain-quantized"],
        tokenizer=tokenizer,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        augmentation_probability=cfg.train.augmentation_probability,
        overfit_single_batch=cfg.train.overfit_single_batch,
    )

    # logger
    wandb.init(project="masked-midi-modelling", name=cfg.logger.run_name, dir=cfg.paths.log_dir)

    device = torch.device(cfg.train.device)

    # model
    roberta_config = RobertaConfig(vocab_size=tokenizer.vocab_size)
    model = RobertaForMaskedLM(roberta_config).to(device)

    # setting up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # load checkpoint if specified in cfg
    if cfg.paths.load_ckpt_path is not None:
        checkpoint = torch.load(cfg.paths.load_ckpt_path)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # checkpoint save path
    num_params_millions = sum([p.numel() for p in model.parameters()]) / 1_000_000
    save_path = f"{cfg.paths.save_ckpt_dir}/{cfg.logger.run_name}-params-{num_params_millions:.2f}M.ckpt"

    # step counts for logging to wandb
    step_count = 0

    for epoch in range(cfg.train.num_epochs):
        # train epoch
        model.train()
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False)
        loss_epoch = 0.0

        for batch_idx, batch in train_loop:
            # metrics returns loss and additional metrics if specified in step function
            loss = forward_step(model, batch, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loop.set_postfix(loss=loss.item())

            step_count += 1
            loss_epoch += loss.item()

            if (batch_idx + 1) % cfg.logger.log_every_n_steps == 0:
                # log metrics
                wandb.log({"train/loss": loss.item()}, step=step_count)

                # save model and optimizer states
                save_checkpoint(model, optimizer, cfg, save_path=save_path)

        training_metrics = {"train/loss_epoch": loss_epoch / len(train_dataloader)}

        model.eval()

        # val epoch
        val_metrics = validation_epoch(
            model,
            val_dataloader,
            device,
        )
        val_metrics = {"val/" + key: value for key, value in val_metrics.items()}

        # maestro test epoch
        test_metrics = validation_epoch(
            model,
            maestro_test,
            device,
        )
        test_metrics = {"maestro/" + key: value for key, value in test_metrics.items()}

        metrics = training_metrics | val_metrics | test_metrics
        wandb.log(metrics, step=step_count)

    # save model at the end of training
    save_checkpoint(model, optimizer, cfg, save_path=save_path)

    wandb.finish()

    # upload model to huggingface if specified in cfg
    if cfg.paths.hf_repo_id is not None:
        upload_to_huggingface(save_path, cfg)


if __name__ == "__main__":
    train()
