# DETR From Scratch

A from-scratch implementation of **Detection Transformer (DETR)** - an end-to-end object detection framework that treats detection as a set prediction problem. This implementation is built with PyTorch and trained on the CARLA synthetic dataset.

> **Paper**: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12956) (Facebook AI Research, ICCV 2020)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Training](#training)
- [Visualization](#visualization)
- [Loss Analysis](#loss-analysis)
- [Model Components](#model-components)
- [Troubleshooting](#troubleshooting)

---

## Overview

DETR fundamentally changes the object detection paradigm by:

1. **Eliminating hand-crafted components** (NMS, anchor boxes) used in traditional detectors
2. **Treating detection as a set prediction problem** using a transformer encoder-decoder
3. **Direct bounding box prediction** without post-processing
4. **Auxiliary losses** from intermediate decoder layers for better convergence

### Key Innovation

Instead of dense predictions, DETR:
- Takes an image as input
- Encodes it with a CNN backbone
- Passes through a transformer encoder-decoder
- Predicts a **fixed set of N queries** (default: 100)
- Each query represents a potential object
- Outputs class + bounding box per query

---

## Key Features

✅ **Full DETR Implementation**
- ResNet-101 backbone with Dilated Convolution (DC5)
- Multi-head self/cross-attention transformer (6 encoder + 6 decoder layers)
- Sine positional encodings
- Auxiliary losses from intermediate decoder outputs

✅ **Advanced Training Features**
- Hungarian algorithm for bipartite matching
- Deep supervision with auxiliary losses
- Gradient clipping for stability
- Separate learning rates for backbone and transformer
- Learning rate scheduling with step decay

✅ **Comprehensive Monitoring**
- TensorBoard integration for loss tracking
- Real-time 2D bounding box visualization
- Per-loss component logging
- Epoch-level and batch-level statistics

✅ **Production Ready**
- Flexible YAML configuration
- Checkpoint management
- Mixed precision training support (experimental)
- DataLoader optimization with prefetching

---

## Architecture

### Component Pipeline

```
Input Image
    ↓
┌─────────────────────────────────────┐
│  ResNet-101 Backbone (w/ DC5)       │ ← Feature extraction
│  Output: C5 features (2048 channels)│
└──────────────┬──────────────────────┘
               ↓
        ┌──────────────┐
        │ 1x1 Conv    │ ← Reduce to hidden_dim (256)
        └──────┬───────┘
               ↓
    ┌──────────────────────┐
    │ Positional Encoding  │ ← Sine embeddings (128D)
    └──────────┬───────────┘
               ↓
    ┌──────────────────────┐
    │ Transformer Encoder  │ ← 6 layers of self-attention
    │ (8 heads, 2048 FFN)  │
    └──────────┬───────────┘
               ↓
    ┌──────────────────────┐
    │ Transformer Decoder  │ ← 6 layers + 100 object queries
    │ (8 heads, 2048 FFN)  │
    └──────────┬───────────┘
               ↓
    ┌──────────────────────┐
    │ Classification Head  │ ← Output: (100, 9) - 8 classes + background
    │ Bbox Regression Head │ ← Output: (100, 4) - normalized [cx, cy, w, h]
    └──────────────────────┘
```

### Key Dimensions

| Component | Dimension | Notes |
|-----------|-----------|-------|
| Input Image | [256, 688] | Resized for CARLA dataset |
| Backbone Output | 2048 channels | ResNet-101 C5 block |
| Hidden Dimension | 256 | Transformer d_model |
| Num Attention Heads | 8 | d_model / 8 = 32 per head |
| FFN Dimension | 2048 | 8x expansion in feedforward |
| Num Queries | 100 | Object query tokens |
| Num Classes | 8 | + 1 background = 9 total |

---

## Project Structure

```
detr_from_scratch/
├── cfg/
│   └── detr.yaml                 # Complete configuration
├── dataset/
│   ├── carla_dataset.py          # CARLA dataset loader
│   └── generic_dataset.py        # Generic dataset interface
├── model/
│   ├── __init__.py               # Module exports
│   ├── backbone.py               # ResNet backbone with DC5
│   ├── detr.py                   # Main DETR module
│   ├── layers.py                 # Transformer components
│   ├── loss.py                   # Loss computation
│   ├── matcher.py                # Hungarian matcher
│   └── positional_encoding.py    # Positional embeddings
├── train.py                      # Training script with visualization
├── utils.py                      # Utilities (config, data processing)
├── box_ops.py                    # Bounding box operations
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

### Model Module Details

**backbone.py**
- ResNet-101 with optional dilated convolution (DC5)
- FrozenBatchNorm for better stability
- Configurable return layers

**detr.py**
- Main DETR class combining all components
- Forward pass: image + mask → detection outputs
- Auxiliary output handling

**layers.py**
- TransformerEncoder (6 layers of self-attention)
- TransformerDecoder (6 layers of cross-attention with queries)
- TransformerEncoderLayer & TransformerDecoderLayer
- MLP head for classification and regression

**loss.py**
- DETRLoss class computing weighted combination of:
  - Cross-entropy loss (classification)
  - L1 loss (bbox regression)
  - GIoU loss (bbox alignment)
- Auxiliary loss support for intermediate decoder outputs
- Class-aware weighting with EOS coefficient

**matcher.py**
- HungarianMatcher using scipy's linear_sum_assignment
- Bipartite matching for prediction-target assignment
- Configurable cost weights

**positional_encoding.py**
- Sine positional encoding (learnable optional)
- 2D positional embeddings for image coordinates

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 8GB+ VRAM recommended

### Setup

```bash
# Clone repository
git clone <repo_url>
cd detr_from_scratch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=1.9.0
torchvision>=0.10.0
scipy>=1.5.0          # For Hungarian algorithm
pyyaml>=5.4           # For config loading
tensorboard>=2.5.0    # For visualization
matplotlib>=3.3.0     # For bbox visualization
numpy>=1.19.0
tqdm>=4.50.0          # Progress bars
```

---

## Configuration

All training parameters are defined in [cfg/detr.yaml](cfg/detr.yaml).

### Key Configuration Sections

**Dataset**
```yaml
dataset:
  root_dir: path/to/carla/dataset
  img_size: [256, 688]           # Input image size
  normalize_mean: [0.485, 0.456, 0.406]  # ImageNet statistics
  normalize_std: [0.229, 0.224, 0.225]
  num_workers: 2                 # DataLoader workers
```

**Model Architecture**
```yaml
model:
  backbone: "resnet101"          # ResNet variant
  dilation: true                 # DC5 dilation
  hidden_dim: 256                # Transformer dimension
  nheads: 8                       # Attention heads
  enc_layers: 6                   # Encoder layers
  dec_layers: 6                   # Decoder layers
  num_queries: 100                # Object queries
  num_classes: 8                  # Object classes (excl. background)
  return_intermediate_dec: true   # Deep supervision
```

**Training Hyperparameters**
```yaml
training:
  lr: 0.0001                     # Main learning rate
  lr_backbone: 0.00001           # Backbone learning rate (lower for pretrained)
  batch_size: 4                   # Batch size
  epochs: 500                     # Training epochs
  lr_drop: 400                    # LR scheduler step size
  clip_max_norm: 0.1             # Gradient clipping
```

**Loss Weights**
```yaml
loss:
  weight_dict:
    loss_ce: 1.0                 # Classification loss
    loss_bbox: 5.0               # Bounding box L1 loss
    loss_giou: 2.0               # GIoU loss
  eos_coef: 0.1                  # Background class weight
```

**TensorBoard Logging**
```yaml
tensorboard:
  log_loss_freq: 5               # Log loss every N steps
  log_img_freq: 50               # Log visualizations every N steps
```

---

## Training

### Basic Usage

```bash
python train.py --config cfg/detr.yaml
```

### Expected Output

```
Epoch [1/500] | Avg Loss: 45.2341 | Time: 125.3s
Epoch [2/500] | Avg Loss: 42.1523 | Time: 124.8s
...
Saved best model checkpoint: runs/current_time/model_5.pth
Epoch [500/500] | Avg Loss: 2.3456 | Time: 128.1s

Training complete! Logs saved to: runs/current_time
```

### Monitor Training with TensorBoard

```bash
tensorboard --logdir=runs
```

Then open http://localhost:6006 in your browser.

### TensorBoard Metrics

**Scalar Metrics**
- `Train/Total_Loss`: Weighted sum of all losses
- `Train/loss_ce`: Classification loss (per batch)
- `Train/loss_bbox`: Bounding box regression loss
- `Train/loss_giou`: GIoU loss
- `Train/loss_ce_i`: Auxiliary classification loss from layer i
- `Train/loss_bbox_i`: Auxiliary bbox loss from layer i
- `Train/loss_giou_i`: Auxiliary GIoU loss from layer i
- `Train/Epoch_Loss`: Average loss per epoch

**Image Metrics**
- `Visualizations/Sample_0` through `Sample_3`: Ground truth (green) and predictions (red) bounding boxes

---

## Visualization

### Real-time Detection Visualization

During training, 2D bounding box visualizations are logged to TensorBoard:

- **Green boxes**: Ground truth annotations
- **Red dashed boxes**: Model predictions with confidence scores
- **Logged every N batches**: Configurable via `log_img_freq`

### Visualization Details

```python
# Example from TensorBoard
Sample 1 - Green: GT, Red: Pred
├── Ground Truth (Green boxes)
│   └── Class label
├── Predictions (Red dashed boxes)
│   └── Class idx + confidence (e.g., "C2 0.87")
```

The visualization automatically:
- Filters low-confidence predictions (< 0.5)
- Denormalizes images using configured statistics
- Converts normalized coordinates to pixel space
- Handles variable batch padding

---

## Loss Analysis

### Loss Magnitude with Auxiliary Losses

With `return_intermediate_dec: true`, the model produces 6 sets of predictions (1 final + 5 intermediate decoder layers).

**Loss Components Per Layer**

| Component | Untrained | Early Training | Converging | Well-Trained |
|-----------|-----------|----------------|------------|--------------|
| loss_ce | 2.1-2.3 | 1.5-2.0 | 0.3-0.5 | 0.05-0.1 |
| loss_bbox | 0.3-0.5 | 0.2-0.3 | 0.1-0.2 | 0.02-0.05 |
| loss_giou | 1.8-2.0 | 1.2-1.5 | 0.3-0.5 | 0.05-0.1 |

**Weighted Loss Per Layer**
```
loss = 1.0 * loss_ce + 5.0 * loss_bbox + 2.0 * loss_giou
     ≈ 1.0 * 2.2 + 5.0 * 0.4 + 2.0 * 1.9
     ≈ 2.2 + 2.0 + 3.8
     ≈ 8.0 per layer
```

**Total Loss (6 layers)**
```
total_loss = 6 × 8.0 ≈ 48.0 (untrained)
           = 6 × 3.5 ≈ 21.0 (early training)
           = 6 × 0.6 ≈ 3.6  (converging)
           = 6 × 0.1 ≈ 0.6  (well-trained)
```

### Expected Loss Progression

```
Epoch 1-5:    Loss ~40-50  (untrained model, learning basic features)
Epoch 10:     Loss ~30-40  (initial learning signal)
Epoch 50:     Loss ~10-20  (significant progress)
Epoch 100+:   Loss ~3-8    (well-trained, fine-tuning)
Epoch 400+:   Loss ~1-3    (convergence, minimal improvement)
```

**Normal Behavior**
- ✅ Loss starts at 40-50
- ✅ Smooth monotonic decrease (with noise)
- ✅ After 100 epochs: < 10
- ✅ After 400 epochs: < 5

**Red Flags**
- ❌ Loss: > 100 (check data loading)
- ❌ Loss: NaN/Inf (gradient explosion, check learning rates)
- ❌ Loss: not decreasing after 20 epochs (learning rate too low)
- ❌ Loss: oscillating wildly (learning rate too high)

---

## Model Components

### 1. Backbone (ResNet-101)

```python
from model import ResnetBackbone

backbone = ResnetBackbone(
    resnet_model="resnet101",
    return_layers={"layer4": "feats"},
    dcn=True  # Dilated Convolution in C5
)
```

**Features**
- Pretrained ImageNet weights
- Frozen batch normalization
- DC5 (dilated convolution) for higher resolution
- Output: [B, 2048, H/32, W/32]

### 2. Positional Encoding

```python
from model import PositionEmbeddingSine

pos_embd = PositionEmbeddingSine(
    num_pos_feats=128,
    temperature=10000,
    normalize=False
)
```

**Formula**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

### 3. Transformer

```python
from model import Transformer

transformer = Transformer(
    d_model=256,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dropout=0.1,
    return_intermediate_dec=True
)
```

**Encoder**: Self-attention on image features
**Decoder**: Cross-attention between queries and encoded features

### 4. DETR Model

```python
from model import DETR

model = DETR(
    backbone=backbone,
    transformer=transformer,
    pos_embd=pos_embd,
    num_classes=8,
    query_dim=100,
    hidden_dim=256
)

# Forward pass
outputs = model(images, masks)
# outputs = {
#     'obj_class': [B, 100, 9],      # Logits for 8 classes + background
#     'bbox': [B, 100, 4],            # Normalized [cx, cy, w, h]
#     'aux_outputs': [                # Intermediate predictions
#         {'obj_class': ..., 'bbox': ...},
#         ...
#     ]
# }
```

### 5. Loss Function

```python
from model import DETRLoss, HungarianMatcher

matcher = HungarianMatcher(
    cost_class=1.0,
    cost_bbox=5.0,
    cost_giou=2.0
)

criterion = DETRLoss(
    num_classes=8,
    matcher=matcher,
    weight_dict={
        'loss_ce': 1.0,
        'loss_bbox': 5.0,
        'loss_giou': 2.0
    },
    eos_coef=0.1
)

# Compute loss
loss_dict = criterion(outputs, targets)
total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict)
```

**Loss Components**
- **Classification Loss**: Cross-entropy between predicted and ground truth classes
- **L1 Bbox Loss**: Smooth L1 distance between predicted and target boxes
- **GIoU Loss**: Generalized IoU for bounding box alignment
- **EOS Coefficient**: Higher weight for background class (prevents class imbalance)

---

## Training Pipeline Details

### Data Loading

```python
from utils import collate_fn, load_config
from dataset.carla_dataset import CarlaDataset

cfg = load_config('cfg/detr.yaml')
dataset = CarlaDataset(cfg)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn  # Handles variable-sized images
)
```

**Custom Collate Function**
- Pads images to maximum size in batch
- Creates attention masks for padded regions
- Maintains correspondence between images and annotations

### Optimizer & Scheduler

```python
# Separate learning rates for backbone and transformer
optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-5},
    {'params': transformer_params, 'lr': 1e-4}
], weight_decay=1e-4)

# Step learning rate decay at epoch 400
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=400
)
```

**Why Separate LRs?**
- Backbone: Pretrained on ImageNet → lower LR for fine-tuning
- Transformer: Random initialization → higher LR for learning

### Training Loop

```python
for epoch in range(num_epochs):
    model.train()
    
    for batch_idx, batch in enumerate(dataloader):
        # Forward pass
        outputs = model(batch['img'], batch['mask'])
        
        # Compute loss (includes auxiliary losses)
        loss_dict = criterion(outputs, batch)
        total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        
        # Logging
        if global_step % log_freq == 0:
            writer.add_scalar('Train/Total_Loss', total_loss, global_step)
        
        global_step += 1
    
    scheduler.step()  # Decay learning rate
```

---

## Troubleshooting

### Common Issues

**1. Loss is NaN/Inf**
- **Cause**: Gradient explosion or numerical instability
- **Solution**: 
  - Reduce learning rate (try 1e-5)
  - Check data preprocessing (images should be in [0, 1])
  - Enable gradient clipping (already enabled in config)

**2. Loss doesn't decrease**
- **Cause**: Learning rate too high/low, or data loading issue
- **Solution**:
  - Print sample batch to verify data
  - Start with learning rate 1e-4
  - Check that targets format matches code

**3. Out of memory**
- **Cause**: Batch size too large for GPU
- **Solution**:
  - Reduce `batch_size` in config (try 2 or 1)
  - Reduce image size
  - Enable gradient accumulation

**4. Visualizations not appearing in TensorBoard**
- **Cause**: Log directory path or visualization errors
- **Solution**:
  - Check that `log_img_freq` is not too large
  - Verify TensorBoard is reading correct log dir
  - Check console for visualization errors

**5. Data loading is slow**
- **Cause**: CPU bottleneck
- **Solution**:
  - Increase `num_workers` (e.g., 4 or 8)
  - Enable `pin_memory: true` (already enabled)
  - Use `prefetch_factor: 2` (already enabled)

---

## Model Performance

### Expected Metrics (on CARLA)

| Metric | Value |
|--------|-------|
| mAP@0.5 | ~70-75% |
| mAP@0.75 | ~50-60% |
| Average Loss | ~1-2 (after convergence) |
| Training Time | ~500 GPU hours (500 epochs) |

*Note: Metrics depend on dataset diversity and annotation quality*

---

## References

1. **DETR Paper**: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12956)
2. **Transformer Paper**: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
3. **Hungarian Algorithm**: [The Hungarian Algorithm for the Assignment Problem](https://en.wikipedia.org/wiki/Hungarian_algorithm)
4. **GIoU**: [Generalized Intersection over Union](https://arxiv.org/abs/1902.09630)

---

## License

This project is an educational implementation. Refer to original DETR repository for official code.

---

## Author Notes

This implementation prioritizes clarity and educational value. For production use:
- Add data augmentation (CutMix, Mosaic)
- Implement multi-scale training
- Add model distillation
- Optimize inference with ONNX/TorchScript
- Fine-tune hyperparameters for your dataset

Happy training! 🚀
