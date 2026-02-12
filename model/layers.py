import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x, pos, mask=None):
        q = k = x + pos

        # Multi headed attention
        attn_out = self.mha(q, k, x, key_padding_mask=mask)[0]

        # Add norm
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # FFN
        ffn_out = self.ffn(x)

        # Add norm
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src, pos, mask=None):

        output = src
        for layer in self.layers:
            output = layer(output, pos, mask)
        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, tgt, enc_out, pos, query_pos, enc_mask=None, dec_mask=None):
        # Self attention
        q = k = tgt + query_pos
        self_attn_out = self.self_attn(q, k, tgt, attn_mask=dec_mask)[0]

        # Add norm
        tgt = tgt + self.dropout(self_attn_out)
        tgt = self.norm1(tgt)

        # MHA with encoder output
        mha_out = self.mha(
            tgt + query_pos, enc_out + pos, enc_out, key_padding_mask=enc_mask
        )[0]

        # Add norm
        tgt = tgt + self.dropout(mha_out)
        tgt = self.norm2(tgt)

        # FFN
        ffn_out = self.ffn(tgt)

        # Add norm
        tgt = tgt + self.dropout(ffn_out)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Module):

    def __init__(self, num_layers, embed_dim, num_heads, norm=None, dropout=0.1):
        super().__init__()
        self.norm = norm
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(embed_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, tgt, enc_out, pos, query_pos, enc_mask=None, dec_mask=None):

        output = tgt
        for layer in self.layers:
            output = layer(output, enc_out, pos, query_pos, enc_mask, dec_mask)

        if self.norm is not None:
            output = self.norm(output)
        return output.unsqueeze(0)


class Transformer(nn.Module):

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout=0.1,
    ):
        super().__init__()
        self.decoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(
            num_encoder_layers, embed_dim=d_model, num_heads=nhead, dropout=dropout
        )
        self.decoder = TransformerDecoder(
            num_decoder_layers,
            embed_dim=d_model,
            num_heads=nhead,
            norm=self.decoder_norm,
            dropout=dropout,
        )

    def forward(self, x, mask, query_embed, pos_embd):
        x = x.flatten(2).permute(2, 0, 1)
        pos = pos_embd.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        # Repeat embeds for all sample in the batch
        query_embed = query_embed.weight.unsqueeze(1).repeat(1, x.shape[1], 1)

        tgt = torch.zeros_like(query_embed)

        memory = self.encoder(x, pos, mask)

        x = self.decoder(tgt, memory, pos, query_embed, enc_mask=mask)

        return x, memory

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x