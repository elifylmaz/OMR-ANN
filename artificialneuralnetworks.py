from ultralytics import YOLO
import cv2

# YOLO modelini yükle
model = YOLO('best-2.pt')

# Cevap anahtarı
answer_key = {
    1: "b",
    2: "c",
    3: "d",
    4: "a",
    5: "b",
    6: "c",
    7: "d",
    8: "a",
    9: "b",
    10: "c"
}

# VideoCapture ile kamerayı başlat
cap = cv2.VideoCapture(0)

try:
    while True:
        # Kameradan bir kare yakala
        ret, frame = cap.read()
        if not ret:
            break

        # Görseli 90 derece saat yönünde döndür
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Model ile tahmin yap
        results = model.predict(source=rotated_frame)

        # Tahmin sonuçlarını işleme ve görsel üzerinde işaretleme
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                score = scores[i]
                cls = classes[i]
                label = model.names[cls]

                # Görsel üzerinde çerçeve ve etiket çizme
                cv2.rectangle(rotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(rotated_frame, f'{label} {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # İşaretlenmiş görüntüyü göster
        cv2.imshow("YOLO Detection", rotated_frame)

        # 'q' tuşuna basarak çıkış yapma imkanı
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Process interrupted.")

finally:
    # Kaynakları serbest bırak ve pencereleri kapat
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")
