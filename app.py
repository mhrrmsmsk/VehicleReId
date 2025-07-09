import os
import cv2
import torch
import traceback
import msvcrt
import numpy as np
from PIL import Image, ImageGrab
from ultralytics import YOLO

from model.lfft import LFFT
from utils.feature_extractor import extract_feature, cosine_sim

SAVE_DIR = "saved_features"
os.makedirs(SAVE_DIR, exist_ok=True)

def get_next_name():
    existing = [f.replace(".pt", "") for f in os.listdir(SAVE_DIR) if f.endswith(".pt")]
    ids = [int(name.split("_")[-1]) for name in existing if name.startswith("vehicle_")]
    next_id = max(ids, default=0) + 1
    return f"vehicle_{next_id:03d}"

def screen_loop(model, known_features, device):
    print("✅ Sistem ekran görüntüsü ile çalışıyor.")
    print("   'a'  → Yeni araç kaydet")
    print("   's'  → Mevcutlara karşı kontrol et ve varsa sil")
    print("   'q'  → Çıkış")

    yolo_model = YOLO("yolov8n.pt") 
    allowed_classes = {"car", "truck", "bus", "motorcycle"}

    try:
        while True:
            screenshot = ImageGrab.grab()
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            if msvcrt.kbhit():
                key = msvcrt.getch().lower()
                if key == b'q':
                    print("🛑 Çıkılıyor...")
                    break

                results = yolo_model(frame)[0]
                detections = results.boxes
                names = results.names

                found_vehicle = False

                for box in detections:
                    cls_id = int(box.cls)
                    label = names[cls_id]
                    if label not in allowed_classes:
                        continue

                    found_vehicle = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vehicle_img = frame[y1:y2, x1:x2]

                    if vehicle_img.size == 0:
                        continue

                    img_rgb = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    feat = extract_feature(model, pil_img).to(device)

                    if key == b'a':
                        best_sim, best_match = -1.0, None
                        for name, ref_feat in known_features.items():
                            sim = cosine_sim(feat, ref_feat.to(device))
                            if sim > best_sim:
                                best_sim, best_match = sim, name

                        if best_sim > 0.90:
                            print(f"[A] Araç zaten kayıtlı → {best_match} ({best_sim:.2f})")
                        else:
                            new_name = get_next_name()
                            torch.save(feat.cpu(), os.path.join(SAVE_DIR, new_name + ".pt"))
                            known_features[new_name] = feat.cpu()
                            print(f"[A] Yeni araç kaydedildi: {new_name}")

                    elif key == b's':
                        if not known_features:
                            print("[S] Henüz kayıtlı araç yok.")
                        else:
                            best_sim, best_match = -1.0, None
                            for name, ref_feat in known_features.items():
                                sim = cosine_sim(feat, ref_feat.to(device))
                                if sim > best_sim:
                                    best_sim, best_match = sim, name
                            if best_sim > 0.90:
                                print(f"[S] Araç çıkışı → {best_match} ({best_sim:.2f})")
                                del known_features[best_match]
                                os.remove(os.path.join(SAVE_DIR, best_match + ".pt"))
                            else:
                                print("[S] Eşleşme bulunamadı!")

                    torch.cuda.synchronize() if device.type == "cuda" else None
                    break  

                if not found_vehicle:
                    print("🚫 Araç tespit edilemedi!")

    except Exception:
        print("⚠️ Döngü sırasında hata oluştu:")
        traceback.print_exc()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LFFT().to(device)
    model.eval()
    print("Model yüklendi.")

    known_features = {}
    for fname in os.listdir(SAVE_DIR):
        if fname.endswith(".pt"):
            path = os.path.join(SAVE_DIR, fname)
            known_features[fname.replace(".pt", "")] = torch.load(path, map_location="cpu")

    screen_loop(model, known_features, device)
