import torch.nn as nn
from .layers import MLP


class DETR(nn.Module):
    def __init__(
        self,
        backbone,
        pos_embd,
        transformer,
        hidden_dim=512,
        query_dim=100,
        num_classes=8,
    ):
        super(DETR, self).__init__()
        self.backbone = backbone
        self.pos_embd = pos_embd
        self.transformer = transformer
        self.conv_block = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(query_dim, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, x, mask):

        features = self.backbone(x, mask)
        x = features["feats"]
        mask = features["mask"]
        pos = self.pos_embd(x, mask)
        x = self.conv_block(x)

        hs, memory = self.transformer(x, mask, self.query_embed, pos)

        obj_class = self.class_embed(hs)
        bbox = self.bbox_embed(hs).sigmoid()

        if self.transformer.decoder.return_intermediate:
            aux_outputs = [
                {"obj_class": a, "bbox": b}
                for a, b in zip(obj_class[:-1], bbox[:-1])
            ]
            return dict(
                obj_class=obj_class[-1], bbox=bbox[-1], memory=memory, aux_outputs=aux_outputs
            )

        return dict(obj_class=obj_class[-1], bbox=bbox[-1], memory=memory)
