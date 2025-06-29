import os
import cv2
import torch
import traceback
import msvcrt
from PIL import Image

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

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Görüntü alınamadı, çıkılıyor.")
                break

            # Konsolda tuş bekle (non-blocking)
            if msvcrt.kbhit():
                key = msvcrt.getch().lower()
                if key == b'q':
                    print("🛑 Çıkılıyor...")
                    break

                # RGB → PIL
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)

                # Özelliği çıkar ve GPU'ya taşı
                feat = extract_feature(model, pil_img).to(device)

                if key == b'a':
                    # Yeni araç kaydet
                    new_name = get_next_name()
                    torch.save(feat.cpu(), os.path.join(SAVE_DIR, new_name + ".pt"))
                    known_features[new_name] = feat.cpu()
                    print(f"[A] Yeni araç kaydedildi: {new_name}")

                elif key == b's':
                    # Mevcutlara karşı benzerliği hesapla
                    if not known_features:
                        print("[S] Henüz kayıtlı araç yok.")
                    else:
                        best_sim, best_match = -1.0, None
                        for name, ref_feat in known_features.items():
                            sim = cosine_sim(feat, ref_feat.to(device))
                            if sim > best_sim:
                                best_sim, best_match = sim, name
                        print(f"[S] En iyi eşleşme → {best_match} ({best_sim:.2f})")

                # küçük debounce
                torch.cuda.synchronize() if device.type=="cuda" else None

    except Exception:
        print("⚠️ Döngü sırasında hata oluştu:")
        traceback.print_exc()
    finally:
        cap.release()
        print("✅ Kamera kapatıldı.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modeli yükle
    model = LFFT().to(device)
    model.eval()
    print("Model yüklendi.")

    # Kayıtlı özellikleri yükle
    known_features = {}
    for fname in os.listdir(SAVE_DIR):
        if fname.endswith(".pt"):
            path = os.path.join(SAVE_DIR, fname)
            known_features[fname.replace(".pt", "")] = torch.load(path, map_location="cpu")

    # Kamerayı başlat
    camera_loop(model, known_features, device)
