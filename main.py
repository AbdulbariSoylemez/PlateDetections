# Gerekli kütüphaneleri ve modülleri yükle
from ultralytics import YOLO
import cv2
import numpy as np

# SORT (Simple Online and Realtime Tracking) algoritmasını içe aktar
from sort.sort import *

# Özel yardımcı fonksiyonları içe aktar
from util import get_car, read_license_plate, write_csv

# Sonuçları depolamak için boş bir sözlük başlat
results = {}

# SORT izleyiciyi başlat
mot_tracker = Sort()

# Araçları tespit etmek için YOLO modelini yükle
coco_model = YOLO("yolov8n.pt")

# Plakaları tespit etmek için YOLO modelini yükle
license_plate_detector = YOLO("/Users/abdulbarisoylemez/Documents/Visual Code/PlakaTespiti/model/license_plate_detector.pt")

# Video dosyasının yolunu belirt
video = cv2.VideoCapture("/Users/abdulbarisoylemez/Documents/Visual Code/PlakaTespiti/sample.mp4")

# Algılanacak araç sınıflarını belirle (örneğin, araba, kamyon)
vehicles = [2, 3, 5, 7]

# Video karelerini döngüye al
frame_nmr = -1
ret = True
while ret:
    # Videodan bir sonraki kareyi oku
    frame_nmr += 1
    ret, frame = video.read()

    if ret:
        # Şu anki kare için sonuçları depola
        results[frame_nmr] = {}

        # YOLO modelini kullanarak karedeki araçları tespit et
        detections = coco_model(frame)[0]
        detections_ = []

        # Araç olmayan tespitleri filtrele ve ilgili bilgileri depola
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # SORT algoritmasını kullanarak araçları takip et
        track_ids = mot_tracker.update(np.asarray(detections_))

        # YOLO modelini kullanarak karedeki plakaları tespit et
        licanse_plates = license_plate_detector(frame)[0]

        # Her tespit edilen plakayı işle
        for licanse_plate in licanse_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = licanse_plate

            # Plakayı takip edilen bir araca ata
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(licanse_plate, track_ids)

            if car_id != -1:
                # Plaka bölgesini kırp
                licanse_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                # Plakayı işle
                licanse_plates_crop_gray = cv2.cvtColor(licanse_plate_crop, cv2.COLOR_BGR2GRAY)
                _, licanse_plates_crop_thresh = cv2.threshold(licanse_plates_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Lisans plakası numarasını oku
                license_plate_text, license_plate_text_score = read_license_plate(licanse_plates_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {
                        "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                        "license_plate": {
                            "bbox": [x1, y1, x2, y2],
                            "text": license_plate_text,
                            "bbox_score": score,
                            "text_score": license_plate_text_score
                        }
                    }

# Sonuçları CSV dosyasına yaz
write_csv(results, "./test.csv")
