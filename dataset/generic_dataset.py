import torch
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
from torchvision.transforms import transforms

class GenericDataset(Dataset):
    """CARLA synthetic dataset loader for unsupervised depth and semantic segmentation.
    
    Loads RGB images from multiple temporal frames, depth maps, semantic masks,
    and camera calibration matrices. Supports multi-scale processing.
    """
    
    def __init__(self, cfg):
        """Initialize CARLA dataset.
        
        Args:
            cfg (SimpleNamespace): Configuration object with virtual_dataset,
                geometry, and other training settings.
        """
        self.cfg = cfg
        root_dir = cfg.dataset.root_dir
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.samples = glob.glob(f"{root_dir}/*.jpg")

    def __len__(self):
        """Return number of valid samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a data sample containing RGB images, depth, semantic labels, and calibration.
        
        Args:
            idx (int): Sample index.
        
        Returns:
            dict: Dictionary with keys for RGB frames at different scales,
                  depth maps, semantic labels, and calibration matrices.
        """
        paths = self.samples[idx]

        # RGB image

        img = self.transforms(Image.open(paths).convert("RGB"))
        # depth = self.load_depth(os.path.join(paths["base_name"], "depth", f"{paths['frame_id']}.npy"))
 
        return img