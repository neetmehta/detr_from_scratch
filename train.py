import torch
from dataset.carla_dataset import CarlaDataset
from torch.utils.data import DataLoader

from utils import load_config
from model import (
    DETR,
    PositionEmbeddingSine,
    ResnetBackbone,
    Transformer,
    HungarianMatcher,
    DETRLoss,
)

import torch.nn.functional as F
import torch.optim as optim
import argparse
from tqdm import tqdm
import time
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os


class Trainer:

    def __init__(self, cfg_path):
        cfg = load_config(cfg_path)
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        device = cfg.runtime.device

        # --- Model Initialization ---
        backbone = ResnetBackbone(
            resnet_model=cfg.model.backbone.resnet_model,
            return_layers=cfg.model.backbone.return_layers,
            dcn=cfg.model.backbone.dcn,
        )
        transformer = Transformer(
            d_model=cfg.model.transformer.d_model,
            nhead=cfg.model.transformer.nhead,
            num_encoder_layers=cfg.model.transformer.num_encoder_layers,
            num_decoder_layers=cfg.model.transformer.num_decoder_layers,
            dropout=cfg.model.transformer.dropout,
        )
        pos_embd = PositionEmbeddingSine(
            num_pos_feats=cfg.model.positional_embedding.num_pos_feats,
            temperature=cfg.model.positional_embedding.temperature,
        )
        self.model = DETR(
            backbone=backbone,
            num_classes=cfg.model.num_classes,
            num_queries=cfg.model.num_queries,
            positional_embedding=pos_embd,
            transformer=transformer,
        ).to(device)

        # Loss functions
        matcher = HungarianMatcher(
            cost_class=cfg.loss.matcher_cost_class,
            cost_bbox=cfg.loss.matcher_cost_bbox,
            cost_giou=cfg.loss.matcher_cost_giou,
        )
        self.criterion = DETRLoss(
            matcher=matcher,
            weight_dict=cfg.loss.weight_dict,
            eos_coef=cfg.loss.eos_coef,
        )

        # --- Optimizer ---
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": cfg.training.lr_backbone,
            },
        ]

        self.optimizer = torch.optim.AdamW(param_dicts, lr=cfg.training.lr,
                                  weight_decay=cfg.training.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, cfg.training.lr_drop)
        # --- Data ---
        self.device = device
        dataset = CarlaDataset(cfg)
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=cfg.training.shuffle,
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory,
            drop_last=cfg.training.drop_last,
        )
        self.epoch = cfg.training.epochs

        # --- TensorBoard Setup ---
        # Defaults to 'runs/current_time' if not specified in config
        log_dir = getattr(cfg.training, "log_dir", None)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.model_out = self.writer.log_dir

        # Frequencies
        self.log_loss_freq = cfg.tensorboard.log_loss_freq  # Log loss every 50 steps
        self.log_img_freq = cfg.tensorboard.log_img_freq  # Log images every 500 steps

    def train(self):
        num_batches = len(self.dataloader)
        global_step = 0
        self.lr_scheduler.step()

        for epoch in range(self.epoch):
            self.model.train()

            epoch_loss = 0.0
            start_time = time.time()

            pbar = tqdm(
                self.dataloader,
                desc=f"Epoch [{epoch+1}/{self.epoch}]",
                dynamic_ncols=True,
            )
            previous_loss = float("inf")

            for batch_idx, inputs in enumerate(pbar):
                iter_start = time.time()

                # Move inputs to device
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(self.device)

                # --- Forward Pass ---
                src, mask = inputs['img'], inputs['mask']

                # --- Backward Pass ---
                self.optimizer.zero_grad()
                outputs = self.model(src, mask)
                total_loss = self.criterion(outputs, inputs)
                total_loss.backward()
                self.optimizer.step()

                # --- Stats ---
                epoch_loss += total_loss.item()
                iter_time = time.time() - iter_start
                samples_per_sec = self.batch_size / iter_time
                global_step += 1

                pbar.set_postfix(
                    {
                        "loss": f"{total_loss.item():.4f}",
                        "avg_loss": f"{epoch_loss / (batch_idx + 1):.4f}",
                        "s/s": f"{samples_per_sec:.1f}",
                    }
                )

                # --- TensorBoard Logging ---
                # if global_step % self.log_loss_freq == 0:
                #     self.writer.add_scalar(
                #         "Train/Total_Loss", total_loss.item(), global_step
                #     )

                # if global_step % self.log_img_freq == 0:
                #     self.log_visuals(inputs, depth_outputs, global_step)

            # End of Epoch
            epoch_time = time.time() - start_time
            print(
                f"\nEpoch {epoch+1}/{self.epoch} | "
                f"Avg Loss: {epoch_loss / num_batches:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            if (epoch_loss / num_batches) < previous_loss:
                previous_loss = epoch_loss / num_batches
                state_dict = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,}
                torch.save(
                    state_dict, os.path.join(self.model_out, f"model_{epoch}.pth")
                )

        self.writer.close()

    
    def log_visuals(self, data, outputs, step):
        """Logs RGB, Reconstructions, and Colormapped Depth to TensorBoard"""
        pass
    

def main():
    parser = argparse.ArgumentParser(description="Monodepth2-style Trainer")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    trainer = Trainer(cfg_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
