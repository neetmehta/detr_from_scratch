import torch
from dataset.carla_dataset import CarlaDataset
from torch.utils.data import DataLoader

from utils import load_config, collate_fn
from model import (
    DETR,
    PositionEmbeddingSine,
    ResnetBackbone,
    Transformer,
    HungarianMatcher,
    DETRLoss,
)
import argparse
from tqdm import tqdm
import time

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms
import numpy as np
import os


class Trainer:

    def __init__(self, cfg_path):
        cfg = load_config(cfg_path)
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        device = cfg.runtime.device

        # --- Model Initialization ---
        # Initialize ResNet backbone with dilation
        return_layers = {cfg.model.return_layers: "feats"}
        backbone = ResnetBackbone(
            resnet_model=cfg.model.backbone,
            return_layers=return_layers,
            dcn=cfg.model.dilation,
        )
        
        # Initialize Transformer with encoder/decoder layers
        transformer = Transformer(
            d_model=cfg.model.hidden_dim,
            nhead=cfg.model.nheads,
            num_encoder_layers=cfg.model.enc_layers,
            num_decoder_layers=cfg.model.dec_layers,
            dropout=cfg.model.dropout,
            return_intermediate_dec=cfg.model.return_intermediate_dec,
        )
        
        # Initialize positional encoding
        pos_embd = PositionEmbeddingSine(
            num_pos_feats=cfg.model.positional_embedding.num_pos_feats,
            temperature=cfg.model.positional_embedding.temperature,
            normalize=cfg.model.positional_embedding.normalize,
        )
        
        # Initialize DETR model
        self.model = DETR(
            backbone=backbone,
            num_classes=cfg.model.num_classes,
            query_dim=cfg.model.num_queries,
            pos_embd=pos_embd,
            transformer=transformer,
            hidden_dim=cfg.model.hidden_dim,
        ).to(device)
        
        if cfg.training.load_pretrained:
            state_dict = torch.load(cfg.training.pretrained_weights_path, map_location=device)
            for k,v in state_dict.items():
                if 'class_embed' in k:
                    print(f"Skipping loading {k} due to shape mismatch")
                    state_dict[k] = self.model.state_dict()[k]
                elif 'bbox_embed' in k:
                    print(f"Skipping loading {k} due to shape mismatch")
                    state_dict[k] = self.model.state_dict()[k]
                    
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {cfg.training.pretrained_weights_path}")

        # --- Loss Function Setup ---
        # Initialize Hungarian Matcher for bipartite matching
        matcher = HungarianMatcher(
            cost_class=cfg.loss.matcher_cost_class,
            cost_bbox=cfg.loss.matcher_cost_bbox,
            cost_giou=cfg.loss.matcher_cost_giou,
        )
        
        # Create weight dictionary for loss calculation
        self.weight_dict = {
            "loss_ce": cfg.loss.weight_dict.loss_ce,
            "loss_bbox": cfg.loss.weight_dict.loss_bbox,
            "loss_giou": cfg.loss.weight_dict.loss_giou,
        }
        
        # Add auxiliary losses for intermediate decoder outputs
        if cfg.model.return_intermediate_dec:
            for i in range(cfg.model.dec_layers - 1):
                self.weight_dict[f"loss_ce_{i}"] = self.weight_dict["loss_ce"]
                self.weight_dict[f"loss_bbox_{i}"] = self.weight_dict["loss_bbox"]
                self.weight_dict[f"loss_giou_{i}"] = self.weight_dict["loss_giou"]
        
        # Initialize DETR loss criterion
        self.criterion = DETRLoss(
            num_classes=cfg.model.num_classes,
            matcher=matcher,
            weight_dict=self.weight_dict,
            eos_coef=cfg.loss.eos_coef,
        )

        # --- Optimizer Setup ---
        # Separate learning rates for backbone and other components
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "backbone" not in n and "transformer" not in n and "class_embed" not in n and "bbox_embed" not in n and p.requires_grad
                ],
                "lr": cfg.training.lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": cfg.training.lr_backbone,
            },
                        {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "transformer" in n and p.requires_grad
                ],
                "lr": cfg.training.lr_transformer,
            },
                                    {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "class_embed" in n and p.requires_grad
                ],
                "lr": cfg.training.lr_class_embed,
            },
                        {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "bbox_embed" in n and p.requires_grad
                ],
                "lr": cfg.training.lr_bbox_embed,
            },
        ]

        self.optimizer = torch.optim.AdamW(
            param_dicts, 
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        
        # Learning rate scheduler: step decay
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=cfg.training.lr_drop,
        )
        
        # --- Data Loading ---
        self.device = device
        dataset = CarlaDataset(cfg)
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=cfg.training.shuffle,
            num_workers=cfg.dataset.num_workers,
            pin_memory=cfg.training.pin_memory,
            drop_last=cfg.training.drop_last,
            collate_fn=collate_fn,
        )
        self.num_epochs = cfg.training.epochs

        # --- TensorBoard Setup ---
        # Defaults to 'runs/current_time' if not specified in config
        log_dir = getattr(cfg.training, "log_dir", None)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.model_out = self.writer.log_dir

        # --- Logging Frequencies ---
        self.log_loss_freq = cfg.tensorboard.log_loss_freq       # Log loss every N steps
        self.log_img_freq = cfg.tensorboard.log_img_freq         # Log images every N steps
        self.start_epoch = 0
        if cfg.training.resume:
            checkpoint_path = cfg.training.checkpoint_path
            if os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location=device)
                self.model.load_state_dict(state_dict['model'], strict=True)
                self.optimizer.load_state_dict(state_dict['optimizer'])
                start_epoch = state_dict.get('epoch', 0) + 1
                print(f"Resumed training from checkpoint: {checkpoint_path} at epoch {start_epoch}")
            else:
                print(f"No checkpoint found at {checkpoint_path}. Starting fresh training.")

    def train(self):
        num_batches = len(self.dataloader)
        global_step = 0
        best_epoch_loss = float("inf")  # Track best model across all epochs
        
        # Initial learning rate step
        self.lr_scheduler.step()

        for epoch in range(self.start_epoch, self.num_epochs):
            # Set model to training mode
            self.model.train()

            epoch_loss = 0.0
            start_time = time.time()

            # Progress bar for the epoch
            pbar = tqdm(
                self.dataloader,
                desc=f"Epoch [{epoch+1}/{self.num_epochs}]",
                dynamic_ncols=True,
            )

            for batch_idx, inputs in enumerate(pbar):
                iter_start = time.time()

                # Move inputs to device
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(self.device)

                # --- Forward Pass ---
                src, mask = inputs['img'], inputs['mask']
                outputs = self.model(src, mask)

                # --- Loss Computation & Backward Pass ---
                self.optimizer.zero_grad()
                loss_dict = self.criterion(outputs, inputs)
                
                # Weighted sum of all losses
                total_loss = sum(
                    loss_dict[k] * self.weight_dict[k] 
                    for k in loss_dict.keys() 
                    if k in self.weight_dict
                )
                
                total_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.cfg.training.clip_max_norm
                )
                
                self.optimizer.step()

                # --- Statistics & Monitoring ---
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
                if global_step % self.log_loss_freq == 0:
                    self.writer.add_scalar(
                        "Train/Total_Loss", total_loss.item(), global_step
                    )
                    for loss_name, loss_value in loss_dict.items():
                        if loss_name in self.weight_dict:
                            self.writer.add_scalar(
                                f"Train/{loss_name}", loss_value.item(), global_step
                            )

                if global_step % self.log_img_freq == 0:
                    self.log_visuals(inputs, outputs, global_step)

            # --- End of Epoch Logging ---
            epoch_time = time.time() - start_time
            avg_epoch_loss = epoch_loss / num_batches
            
            print(
                f"\nEpoch {epoch+1}/{self.num_epochs} | "
                f"Avg Loss: {avg_epoch_loss:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Log to TensorBoard
            self.writer.add_scalar("Train/Epoch_Loss", avg_epoch_loss, epoch)
            
            # Save best model checkpoint
            if avg_epoch_loss < best_epoch_loss:
                best_epoch_loss = avg_epoch_loss
                state_dict = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": avg_epoch_loss,
                }
                checkpoint_path = os.path.join(self.model_out, f"model_{epoch}.pth")
                torch.save(state_dict, checkpoint_path)
                print(f"Saved best model checkpoint: {checkpoint_path}")

            # Step learning rate scheduler at end of epoch
            self.lr_scheduler.step()

        # Close TensorBoard writer
        self.writer.close()
        print(f"\nTraining complete! Logs saved to: {self.model_out}")

    
    def log_visuals(self, inputs, outputs, step):
        """Logs images with 2D bounding boxes to TensorBoard.
        
        Visualizes ground truth and predicted bounding boxes overlaid on images.
        
        Args:
            inputs (dict): Input batch containing 'img', 'labels', 'boxes'
            outputs (dict): Model outputs containing 'obj_class' and 'bbox'
            step (int): Global training step for TensorBoard logging
        """
        try:
            # Get batch size
            batch_size = inputs['img'].shape[0]
            num_samples = min(batch_size, 4)  # Visualize first 4 samples
            
            # Get normalization stats from config
            norm_mean = torch.tensor(self.cfg.dataset.normalize_mean).view(3, 1, 1).to(self.device)
            norm_std = torch.tensor(self.cfg.dataset.normalize_std).view(3, 1, 1).to(self.device)
            
            fig_list = []
            
            for sample_idx in range(num_samples):
                # Get image and denormalize
                img = inputs['img'][sample_idx].clone()
                # img = img * norm_std + norm_mean
                img = torch.clamp(img, 0, 1)
                
                # Get image dimensions
                _, img_h, img_w = img.shape
                
                # Convert to numpy for matplotlib
                img_np = img.permute(1, 2, 0).cpu().numpy()
                
                # Create figure
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                ax.imshow(img_np)
                
                # --- Draw Ground Truth Boxes ---
                gt_boxes = inputs['boxes'][sample_idx]
                gt_labels = inputs['labels'][sample_idx]
                
                for box, label in zip(gt_boxes, gt_labels):
                    # Convert from [xc, yc, w, h] normalized to pixel coordinates
                    xc, yc, w, h = box
                    x1 = (xc - w/2) * img_w
                    y1 = (yc - h/2) * img_h
                    width = w * img_w
                    height = h * img_h
                    
                    # Draw rectangle
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=2, edgecolor='green', facecolor='none',
                        label='Ground Truth'
                    )
                    ax.add_patch(rect)
                    
                    # Add label text
                    class_name = f"Class {label.item()}"
                    ax.text(x1, y1 - 5, class_name, color='green', fontsize=8,
                           bbox=dict(facecolor='green', alpha=0.3))
                
                # --- Draw Predicted Boxes ---
                pred_probs = torch.softmax(outputs['obj_class'][sample_idx], dim=-1)
                pred_boxes = outputs['bbox'][sample_idx]
                
                # Filter predictions by confidence (class != background, i.e., not last class)
                bg_class = self.cfg.model.num_classes
                pred_conf = pred_probs[:, :bg_class].max(dim=-1)  # Max confidence excl. background
                
                # Keep boxes with confidence > 0.5
                conf_threshold = 0.5
                valid_mask = pred_conf.values > conf_threshold
                
                for pred_idx in range(pred_boxes.shape[0]):
                    if not valid_mask[pred_idx]:
                        continue
                    
                    # Get box and class
                    box = pred_boxes[pred_idx]
                    class_idx = pred_conf.indices[pred_idx].item()
                    confidence = pred_conf.values[pred_idx].item()
                    
                    # Convert from [xc, yc, w, h] normalized to pixel coordinates
                    xc, yc, w, h = box
                    x1 = (xc - w/2) * img_w
                    y1 = (yc - h/2) * img_h
                    width = w * img_w
                    height = h * img_h
                    
                    # Draw rectangle
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=2, edgecolor='red', facecolor='none',
                        linestyle='--', label='Prediction'
                    )
                    ax.add_patch(rect)
                    
                    # Add label text with confidence
                    class_name = f"C{class_idx} {confidence:.2f}"
                    ax.text(x1, y1 + 15, class_name, color='red', fontsize=8,
                           bbox=dict(facecolor='red', alpha=0.3))
                
                ax.set_title(f"Sample {sample_idx + 1} - Green: GT, Red: Pred")
                ax.legend(loc='upper right')
                ax.axis('off')
                
                fig_list.append(fig)
            
            # Log all figures to TensorBoard
            for fig_idx, fig in enumerate(fig_list):
                # Convert matplotlib figure to tensor
                fig.canvas.draw()
                img_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                img_array = img_array[:, :, 1:4]  # Drop alpha channel
                img_tensor = transforms.ToTensor()(img_array)
                
                self.writer.add_image(
                    f"Visualizations/Sample_{fig_idx}",
                    img_tensor,
                    global_step=step
                )
                
                plt.close(fig)
            
            print(f"Logged visualization at step {step}")
            
        except Exception as e:
            print(f"Error during visualization logging: {e}")
            import traceback
            traceback.print_exc()
    

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
