import os
import cv2
import torch
import traceback
import msvcrt
from PIL import Image
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

def camera_loop(model, known_features, device):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kamera açılamadı.")
        return

    print("✅ Kamera açıldı.")
    print("   'a'  → Yeni araç kaydet")
    print("   's'  → Mevcutlara karşı kontrol et")
    print("   'q'  → Çıkış")

    # YOLOv8 modelini yükle (car, bus, truck vs. için)
    yolo_model = YOLO("yolov8n.pt")  # En hafif model, istersen yolov8m.pt veya yolov8l.pt kullanabilirsin
    allowed_classes = {"car", "truck", "bus", "motorcycle"}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Görüntü alınamadı, çıkılıyor.")
                break

            if msvcrt.kbhit():
                key = msvcrt.getch().lower()
                if key == b'q':
                    print("🛑 Çıkılıyor...")
                    break

                # Araç tespiti yap
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
                                print(f"[S] En iyi eşleşme → {best_match} ({best_sim:.2f})")
                            else:
                                print("[S] Eşleşme bulunamadı!")

                    torch.cuda.synchronize() if device.type == "cuda" else None
                    break  # Sadece ilk aracı işle

                if not found_vehicle:
                    print("🚫 Araç tespit edilemedi!")

    except Exception:
        print("⚠️ Döngü sırasında hata oluştu:")
        traceback.print_exc()
    finally:
        cap.release()
        print("✅ Kamera kapatıldı.")

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

    camera_loop(model, known_features, device)
