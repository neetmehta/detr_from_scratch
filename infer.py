import torch
import cv2
import numpy as np

class DETRInference:
    
    def __init__(self, model, device, threshold=0.9):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.num_classes = model.num_classes
        self.threshold = threshold

    def do_inference_single_image(self, img: torch.Tensor, mask=None):
        
        img = img.to(self.device)
        assert len(img.shape) == 4, "Please provide batched inputs"
        
        if mask is None:
            b, c, h, w = img.shape
            mask = torch.zeros((b,h,w), dtype=bool, device=self.device)
            
        with torch.no_grad():
            out = self.model(img, mask)
            
        pred_logits, pred_boxes = out['obj_class'].softmax(-1), out['bbox']
        indices = pred_logits.argmax(-1)
        valid_indices = indices != self.num_classes
        not_bg_class = indices[valid_indices]
        not_bg_class_conf = pred_logits[valid_indices, not_bg_class]
        
        high_conf_indices = not_bg_class_conf > self.threshold
        all_valid_class = not_bg_class[high_conf_indices]
        all_valid_boxes_indices = torch.nonzero(valid_indices)[high_conf_indices].transpose(1,0)
        all_valid_boxes = pred_boxes[all_valid_boxes_indices.tolist()]
        return all_valid_class.cpu(), all_valid_boxes.cpu()
    
    def visualize_predictions(self, img, class_names=None):
        """
        Visualize predictions by drawing bounding boxes on the image.
        
        Args:
            img: Input image tensor (B, C, H, W) or numpy array
            classes: Predicted class indices (tensor or numpy array)
            boxes: Predicted bounding boxes in normalized format (N, 4) where format is (x1, y1, x2, y2)
            class_names: List of class names. If None, uses "class 0", "class 1", etc.
        
        Returns:
            numpy array with visualized bounding boxes in uint8 format
        """
        if isinstance(img, np.ndarray):
            # Assume numpy is HWC, RGB. Convert to Tensor (B, C, H, W)
            tensor_img = torch.from_numpy(img).permute(2, 0, 1).float()
            if tensor_img.max() > 1.0: 
                tensor_img /= 255.0
            tensor_img = tensor_img.unsqueeze(0) # Add batch dim
        else:
            tensor_img = img

        # 2. Run Inference
        # Returns: classes (N,), boxes (N, 4) in (cx, cy, w, h) normalized format
        pred_classes, pred_boxes = self.do_inference_single_image(tensor_img)

        # 3. Prepare Image for Visualization (Convert to HWC uint8 numpy)
        # We visualize the first image in the batch
        if isinstance(img, torch.Tensor):
            img_vis = img[0].detach().cpu().permute(1, 2, 0).numpy()
            # Min-Max normalize to 0-255 range for visualization
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-6) * 255
            img_vis = img_vis.astype(np.uint8)
        else:
            # If input was numpy, assume it's ready or scale it
            img_vis = img.copy()
            if img_vis.dtype != np.uint8:
                img_vis = (img_vis * 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        h, w, _ = img_vis.shape

        # 4. Draw Detections
        for cls, box in zip(pred_classes, pred_boxes):
            cls_id = int(cls.item())
            
            # Determine Label
            if class_names and cls_id < len(class_names):
                label = class_names[cls_id]
            else:
                label = f"Class {cls_id}"

            # Convert Box: (cx, cy, w, h) normalized -> (x1, y1, x2, y2) pixel coords
            cx, cy, bw, bh = box.unbind(-1)
            x1 = int((cx - 0.5 * bw) * w)
            y1 = int((cy - 0.5 * bh) * h)
            x2 = int((cx + 0.5 * bw) * w)
            y2 = int((cy + 0.5 * bh) * h)

            # Generate Consistent Color based on Class ID
            np.random.seed(cls_id)
            color = np.random.randint(0, 255, size=3).tolist()

            # Draw Rectangle
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)

            # Draw Label Background and Text
            (t_w, t_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_vis, (x1, y1 - t_h - 5), (x1 + t_w, y1), color, -1)
            cv2.putText(img_vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img_vis