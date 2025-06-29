
import torch
from torchvision import transforms

import torch.nn.functional as F

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def extract_feature(model, image):
    device = next(model.parameters()).device
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(image_tensor)
    return feat.squeeze(0).cpu()

def cosine_sim(f1, f2):
    return F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
