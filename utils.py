import torch

import yaml
import torch
from types import SimpleNamespace


def dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace"""
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)


def resolve_device(device_str: str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda":
        return torch.device("cuda")
    if device_str == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device option: {device_str}")


def load_config(config_path: str):
    """
    Loads YAML config and returns a dot-accessible config object.
    """
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # -------------------------
    # Basic validation
    # -------------------------
    required_sections = [

    ]

    for section in required_sections:
        if section not in cfg_dict:
            raise KeyError(f"Missing config section: {section}")

    # -------------------------
    # Resolve device
    # -------------------------
    device = resolve_device(cfg_dict["runtime"].get("device", "auto"))
    cfg_dict["runtime"]["device"] = device

    # -------------------------
    # Convert to namespace
    # -------------------------
    cfg = dict_to_namespace(cfg_dict)

    return cfg


def create_images_and_masks(tensor_list):
    shape_list = [list(img.shape) for img in tensor_list]

    max_shape = list(shape_list[0])
    for img_shape in shape_list[1:]:
        assert len(img_shape)==3, "Invalid image shape"
        for i, dim_shape in enumerate(img_shape):
            max_shape[i] = max(max_shape[i], dim_shape)
            
    final_shape = [len(tensor_list)] + max_shape
    b, c, h, w = final_shape
    img_tensor = torch.zeros(final_shape)
    mask = torch.ones((b, h, w), dtype=bool)

    for img, pad_img, m in zip(tensor_list, img_tensor, mask):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        m[: img.shape[1], :img.shape[2]] = False
        
    return img_tensor, mask

def collate_fn(samples):
    
    labels = []
    boxes = []
    imgs = []
    for sample in samples:
        imgs.append(sample[0])
        labels.append(sample[1])
        boxes.append(sample[2])
        
    img, mask = create_images_and_masks(imgs)
        
    return dict(img=img, mask=mask, labels=labels, boxes=boxes)
    

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
