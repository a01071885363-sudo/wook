
import cv2
import torch
import serial
import time

# 아두이노 시리얼 연결
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)
arduino.reset_input_buffer()

# YOLOv5 모델 불러오기
model = torch.hub.load('.', 'custom', path='yolov5s.pt', source='local')
model = model.to('cuda:0')
model.conf = 0.5

# USB 카메라 실행
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라 연결 실패")
    arduino.close()
    raise SystemExit

print("객체인식 및 아두이노 전송 시작")

last_sent = ""
last_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for *box, conf, cls in results.xyxy[0].tolist():
            name = model.names[int(cls)].upper()

            # 같은 결과를 계속 전송하지 않도록 제한
            if name != last_sent or time.time() - last_time > 3:
                arduino.write((name + '\n').encode())
                arduino.flush()

                print("JETSON SEND:", name)

                response = arduino.readline().decode(
                    errors='replace'
                ).strip()

                print("ARDUINO:", response)

                last_sent = name
                last_time = time.time()

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

        cv2.imshow("YOLOv5 Arduino Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()
