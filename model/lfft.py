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

        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.embed_dim = self.vit.config.hidden_size

        self.norm = nn.LayerNorm(self.embed_dim)
        self.jspm = JSPM()

    def forward(self, x):
        
        outputs = self.vit(pixel_values=x, output_attentions=True)
        tokens = outputs.last_hidden_state       
        attn_weights = outputs.attentions[-1]    
        B = tokens.size(0)
        tokens = self.norm(tokens)
        local_feats = self.jspm(tokens, attn_weights)
        global_feat = tokens[:, 0]
        return torch.cat([global_feat, local_feats.view(B, -1)], dim=1)