import json
import torch
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
from torchvision.transforms import transforms

IGNORE_LABEL = 255

class CarlaDataset(Dataset):
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
            transforms.ToTensor(), transforms.Resize(cfg.dataset.img_size)
        ])
        self.samples = []
        for scene in glob.glob(os.path.join(root_dir, "run_*")):
            self.samples.extend(self.parse_scenes(scene))

    def parse_scenes(self, scene):
        """Parse scene directory and extract valid frame triplets.
        
        Identifies consecutive frames (t-1, t, t+1) that exist in the dataset.
        
        Args:
            scene (str): Path to scene directory.
        
        Returns:
            list: List of dictionaries with keys 'base_name', 'frame_id', and 'scene'.
        """
        image_paths = [i for i in glob.glob(os.path.join(scene, "image_2", "rgb_images", "*.jpg"))]

        return image_paths
    
    def get_bboxes(self, json_path):
        bboxes_dict = json.load(open(json_path))
        return bboxes_dict

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
        frame_id = int(os.path.normpath(paths).split(os.path.sep)[-1].split(".")[0])
        image_dir = os.path.dirname(paths)
        base_name = os.path.dirname(image_dir)
        # RGB image
        ori_image = Image.open(os.path.join(base_name, "rgb_images", f"{frame_id}.jpg")).convert("RGB")
        W, H = ori_image.size
        
        img = self.transforms(ori_image)
        bboxes = self.get_bboxes(os.path.join(base_name, "2d_bb_labels", f"{frame_id}.json"))
        class_tgt = []
        bbox_tgt = []
        for box in bboxes:
            x1, y1, x2, y2 = box["bbox_2d"]
            xc = ((x2 + x1)/2)/W
            yc = ((y2 + y1)/2)/H
            w = (x2 - x1)/W
            h = (y2 - y1)/H
            class_tgt.append(box["semantic_label"] - 12)
            bbox_tgt.append([xc, yc, w, h])
            
        class_tgt = torch.tensor(class_tgt, dtype=torch.long)
        bbox_tgt = torch.tensor(bbox_tgt, dtype=torch.float32) 
            
        return img, class_tgt, bbox_tgt