import torch
import torch.nn as nn
from transformers import ViTModel

class JSPM(nn.Module):
    def __init__(self, num_groups=4):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x, attn_weights):
        B, N, D = x.size()
        attn_mean = attn_weights.mean(dim=1)
        important_indices = attn_mean.mean(dim=1).topk(self.num_groups, largest=True)[1]
        patches = []
        for i in range(B):
            selected = x[i, important_indices[i]]
            patches.append(selected)
        return torch.stack(patches)

class LFFT(nn.Module):
    def __init__(self):
        super().__init__()
        # Burada doğrudan HuggingFace’den indirip uygun anahtarlarla yüklüyoruz
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.embed_dim = self.vit.config.hidden_size

        self.norm = nn.LayerNorm(self.embed_dim)
        self.jspm = JSPM()

    def forward(self, x):
        # x: [B, 3, 224, 224] normalize edilmiş pixel_values
        outputs = self.vit(pixel_values=x, output_attentions=True)
        tokens = outputs.last_hidden_state       # [B, N, D]
        attn_weights = outputs.attentions[-1]    # [B, H, N, N]

        B = tokens.size(0)
        tokens = self.norm(tokens)

        # JSPM, [B, num_groups, D] dönecek şekilde implement edilmiş olmalı
        local_feats = self.jspm(tokens, attn_weights)

        # CLS token global feature
        global_feat = tokens[:, 0]

        # Final: [B, D + num_groups * D]
        return torch.cat([global_feat, local_feats.view(B, -1)], dim=1)