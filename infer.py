import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os

# Constants
MEANS = (103.94, 116.78, 123.68)
STD = (57.38, 57.12, 58.40)
INPUT_SIZE = 550

class_names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def preprocess(img):
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
    img = (img - MEANS) / STD
    img = img[:, :, ::-1]  # BGR to RGB
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, 0).astype(np.float32)
    return img


def generate_priors():
    feature_map_sizes = [[69, 69], [35, 35], [18, 18], [9, 9], [5, 5]]
    aspect_ratios = [[1, 0.5, 2]] * len(feature_map_sizes)
    scales = [24, 48, 96, 192, 384]
    priors = []

    for idx, fsize in enumerate(feature_map_sizes):
        scale = scales[idx]
        for y in range(fsize[0]):
            for x in range(fsize[1]):
                cx = (x + 0.5) / fsize[1]
                cy = (y + 0.5) / fsize[0]
                for ratio in aspect_ratios[idx]:
                    r = np.sqrt(ratio)
                    w = scale / INPUT_SIZE * r
                    h = scale / INPUT_SIZE / r
                    priors.append([cx, cy, w, h])

    return np.array(priors, dtype=np.float32)


def decode(loc, priors, variances=[0.1, 0.2]):
    boxes = np.zeros_like(loc)
    boxes[:, :2] = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
    boxes[:, 2:] = priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nms(boxes, scores, iou_threshold=0.5):
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold
    )
    return np.array(indices).flatten() if len(indices) > 0 else np.array([], dtype=int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def postprocess(output, original_shape):
    loc, conf, mask, _, proto = output
    loc = np.squeeze(loc, axis=0)
    conf = np.squeeze(conf, axis=0)
    mask = np.squeeze(mask, axis=0)
    proto = np.squeeze(proto, axis=0)

    scores = np.max(conf[:, 1:], axis=1)
    classes = np.argmax(conf[:, 1:], axis=1)
    keep = scores > 0.5

    if not np.any(keep):
        return [], [], [], []

    scores = scores[keep]
    classes = classes[keep]
    mask = mask[keep]
    loc = loc[keep]

    priors = generate_priors()[keep]

    boxes = decode(loc, priors)
    keep_nms = nms(boxes, scores, iou_threshold=0.5)

    boxes = boxes[keep_nms]
    scores = scores[keep_nms]
    classes = classes[keep_nms]
    mask = mask[keep_nms]

    # Generate masks
    masks = proto @ mask.T  # shape: (h, w, N)
    masks = sigmoid(masks)
    masks = np.transpose(masks, (2, 0, 1))  # (N, h, w)

    # Resize to original image size
    resized_masks = []
    for m in masks:
        resized = cv2.resize(m, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        resized_masks.append(resized > 0.5)

    masks = np.array(resized_masks, dtype=bool)

    return masks, classes, scores, boxes

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def infer(engine, image):
    input_data = preprocess(image)
    original_shape = image.shape[:2]  # (H, W)

    with engine.create_execution_context() as context:
        input_shape = (1, 3, INPUT_SIZE, INPUT_SIZE)
        context.set_binding_shape(0, input_shape)

        input_size = np.prod(input_shape)
        output_shapes = [(1, 19248, 4), (1, 19248, 81), (1, 19248, 32), (1, 19248), (1, 32, 138, 138)]
        output_sizes = [np.prod(shape) for shape in output_shapes]
        output_dtypes = [np.float32] * len(output_shapes)

        # Allocate memory
        d_input = cuda.mem_alloc(input_data.nbytes)
        d_outputs = [cuda.mem_alloc(s * np.dtype(dt).itemsize) for s, dt in zip(output_sizes, output_dtypes)]
        bindings = [int(d_input)] + [int(o) for o in d_outputs]

        stream = cuda.Stream()

        # Transfer input
        cuda.memcpy_htod_async(d_input, input_data, stream)

        # Inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Retrieve output
        outputs = []
        for i in range(len(output_shapes)):
            host_mem = np.empty(output_shapes[i], dtype=np.float32)
            cuda.memcpy_dtoh_async(host_mem, d_outputs[i], stream)
            outputs.append(host_mem)

        stream.synchronize()

    return outputs, original_shape


if __name__ == "__main__":
    engine_path = "yolact_yourmodel.engine"
    image_path = "test_images/alovera_1.jpg"

    engine = load_engine(engine_path)
    image = cv2.imread(image_path)

    outputs, orig_shape = infer(engine, image)
    masks, classes, scores, boxes = postprocess(outputs, orig_shape)

    for i, mask in enumerate(masks):
        color = np.random.randint(0, 255, 3).tolist()
        image[mask] = image[mask] * 0.5 + np.array(color, dtype=np.float32) * 0.5

        x1, y1, x2, y2 = boxes[i]
        x1, y1, x2, y2 = map(int, [x1 * orig_shape[1], y1 * orig_shape[0], x2 * orig_shape[1], y2 * orig_shape[0]])
        label = f"{class_names[int(classes[i])]} {scores[i]:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1+5, max(y1 - 5, 0)+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLACT TensorRT Inference", image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
