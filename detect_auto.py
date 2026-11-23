import cv2
import numpy as np
import onnxruntime as ort
import serial
import time

MODEL_PATH = 'best_field.onnx'
PORT = '/dev/ttyACM0'
BAUD = 9600
CLASSES = ['plastic', 'can', 'glass']
COMMANDS = {'plastic': 'P', 'can': 'C', 'glass': 'G'}
CONF_THRES = 0.60
IOU_THRES = 0.45
SIZE = 640
STABLE_FRAMES = 4
CLEAR_FRAMES = 8
ACK_TIMEOUT = 2.0
DONE_TIMEOUT = 10.0


def preprocess(frame):
    h, w = frame.shape[:2]
    scale = min(float(SIZE) / w, float(SIZE) / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh))
    image = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
    px, py = (SIZE - nw) // 2, (SIZE - nh) // 2
    image[py:py + nh, px:px + nw] = resized
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
    image /= 255.0
    return np.expand_dims(image, axis=0), scale, px, py


def nms(boxes, scores, threshold):
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        overlap = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area1 = max(0, boxes[i, 2] - boxes[i, 0]) * max(0, boxes[i, 3] - boxes[i, 1])
        area2 = np.maximum(0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(0, boxes[rest, 3] - boxes[rest, 1])
        iou = overlap / (area1 + area2 - overlap + 1e-6)
        order = rest[iou < threshold]
    return keep


def wait_for_line(port, expected, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = port.readline().decode('utf-8', errors='replace').strip()
        if line:
            print('ARDUINO:', line, flush=True)
            if line == expected:
                return True
            if line == 'UNKNOWN COMMAND':
                return False
    return False


def send_command(port, name):
    cmd = COMMANDS[name]
    port.reset_input_buffer()
    port.write((cmd + '\n').encode('ascii'))
    port.flush()
    print('JETSON SEND:', cmd, '(' + name + ')', flush=True)
    if not wait_for_line(port, 'ACK:' + cmd, ACK_TIMEOUT):
        print('WARNING: ACK not received; do not retry automatically', flush=True)
        return False
    if not wait_for_line(port, 'DONE:' + cmd, DONE_TIMEOUT):
        print('WARNING: DONE not received; do not retry automatically', flush=True)
        return False
    return True


def main():
    print('Loading ONNX model...', flush=True)
    session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    print('Providers:', session.get_providers(), flush=True)
    print('Opening Arduino...', flush=True)
    arduino = serial.Serial(PORT, BAUD, timeout=0.2)
    time.sleep(2.0)
    arduino.reset_input_buffer()
    print('Opening camera...', flush=True)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        arduino.close()
        raise RuntimeError('Camera error')

    candidate = None
    stable = 0
    clear = 0
    locked = False
    print('Ready: show one item; press Q to quit', flush=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('Frame error', flush=True)
                break
            start = time.time()
            image, scale, px, py = preprocess(frame)
            output = session.run(None, {input_name: image})[0][0]
            class_ids = np.argmax(output[:, 5:], axis=1)
            scores = output[:, 4] * np.max(output[:, 5:], axis=1)
            mask = scores >= CONF_THRES
            predictions = output[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]
            best_name = None
            best_score = -1.0
            if len(predictions):
                boxes = predictions[:, :4].copy()
                boxes[:, 0] -= boxes[:, 2] / 2
                boxes[:, 1] -= boxes[:, 3] / 2
                boxes[:, 2] += boxes[:, 0]
                boxes[:, 3] += boxes[:, 1]
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / scale
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / scale
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame.shape[1])
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame.shape[0])
                for cls in np.unique(class_ids):
                    indices = np.where(class_ids == cls)[0]
                    for k in nms(boxes[indices], scores[indices], IOU_THRES):
                        i = indices[k]
                        name = CLASSES[int(cls)]
                        score = float(scores[i])
                        x1, y1, x2, y2 = boxes[i].astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, '{} {:.2f}'.format(name, score),
                                    (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 2)
                        if score > best_score:
                            best_name = name
                            best_score = score

            if best_name is None:
                candidate = None
                stable = 0
                clear += 1
                if locked and clear >= CLEAR_FRAMES:
                    locked = False
                    print('RESET: area clear', flush=True)
            else:
                clear = 0
                if not locked:
                    if best_name == candidate:
                        stable += 1
                    else:
                        candidate = best_name
                        stable = 1
                    if stable >= STABLE_FRAMES:
                        # Lock before sending: never send a second command for this item.
                        locked = True
                        print('CONFIRMED:', best_name, flush=True)
                        send_command(arduino, best_name)

            fps = 1.0 / max(time.time() - start, 0.001)
            cv2.putText(frame, 'FPS: {:.1f}'.format(fps), (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            state = 'WAIT REMOVE ITEM' if locked else 'READY'
            cv2.putText(frame, state, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow('Waste Detection + Arduino', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        arduino.close()


if __name__ == '__main__':
    main()
