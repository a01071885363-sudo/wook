
import cv2
import numpy as np
import onnxruntime as ort
import time

MODEL_PATH = "best.onnx"
CLASSES = ["plastic", "can", "glass"]
CONF_THRES = 0.50
IOU_THRES = 0.45
SIZE = 640

# Load ONNX model
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("Model loaded")
print("Providers:", session.get_providers())

# Open USB camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Camera error")
    raise SystemExit

def preprocess(frame):
    h, w = frame.shape[:2]
    scale = min(SIZE / w, SIZE / h)

    nw = round(w * scale)
    nh = round(h * scale)

    resized = cv2.resize(frame, (nw, nh))
    image = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)

    px = (SIZE - nw) // 2
    py = (SIZE - nh) // 2

    image[py:py + nh, px:px + nw] = resized
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1)
    image = np.ascontiguousarray(image, dtype=np.float32)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    return image, scale, px, py

def nms(boxes, scores, threshold):
    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        rest = order[1:]

        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)

        area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area2 = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])

        iou = inter / (area1 + area2 - inter + 1e-6)
        order = rest[iou < threshold]

    return keep

print("Detection started - press Q to quit")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame error")
            break

        start = time.time()

        image, scale, px, py = preprocess(frame)

        output = session.run(
            None,
            {input_name: image}
        )[0][0]

        class_ids = np.argmax(output[:, 5:], axis=1)
        scores = output[:, 4] * np.max(output[:, 5:], axis=1)

        mask = scores >= CONF_THRES

        predictions = output[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(predictions) > 0:
            boxes = predictions[:, :4].copy()

            boxes[:, 0] -= boxes[:, 2] / 2
            boxes[:, 1] -= boxes[:, 3] / 2
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]

            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / scale

            boxes[:, [0, 2]] = np.clip(
                boxes[:, [0, 2]], 0, frame.shape[1]
            )
            boxes[:, [1, 3]] = np.clip(
                boxes[:, [1, 3]], 0, frame.shape[0]
            )

            for cls in np.unique(class_ids):
                indices = np.where(class_ids == cls)[0]
                kept = nms(boxes[indices], scores[indices], IOU_THRES)

                for k in kept:
                    i = indices[k]
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    name = CLASSES[int(cls)]
                    confidence = scores[i]

                    label = "{} {:.2f}".format(name, confidence)

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame, label, (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2
                    )

        elapsed = time.time() - start
        fps = 1.0 / max(elapsed, 0.001)

        cv2.putText(
            frame, "FPS: {:.1f}".format(fps),
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )

        cv2.imshow("Waste Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
