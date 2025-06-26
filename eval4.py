import onnxruntime
import numpy as np
import cv2

# Constants
MEANS = (103.94, 116.78, 123.68)
STD = (57.38, 57.12, 58.40)
INPUT_SIZE = 550

class_names = ['.', 'Amruthaballi', 'Anthracnose', 'Astma_weed', 'Bamboo', 'CVC',
    'Citrus bacterial spot', 'Citrus greasy spot', 'Hibiscus', 'Jackfruit',
    'Jasmine', 'Kambajala', 'Kohlrabi', 'Lantana', 'Nelavembu', 'Parijatha',
    'Phoma blight', 'alovera', 'alternaria blight', 'amla', 'anthracnosenooni',
    'arali', 'ashoka', 'badipala', 'balloon_Vine', 'bacterial blast',
    'bacterial wilt', 'beans', 'betel', 'bhrami', 'black root rot raddish',
    'botryis blight', 'bringaraja', 'camphor', 'caricature', 'castor',
    'catharanthus', 'chakte', 'chilly', 'citron canker', 'citron lime',
    'citrus anthracnose', 'coffee', 'coffee leaf rust', 'coriender',
    'coriender blight', 'curry', 'defected arali', 'defected betel',
    'defected camphor', 'defectedAmruthaballi', 'defectedBamboo',
    'defectedalovera', 'defectedamla', 'defectedashoka', 'defectedbadipala',
    'defectedballoon_Vine', 'defectedbeans', 'defectedbhrami',
    'defectedbringaraja', 'defectedcatharanthus', 'defectedseethaashoka',
    'defectedspinach', 'defectedtamarind', 'defectedtaro', 'defectedtecoma',
    'defectedthambe', 'defectedtomato', 'defectedtulsi',
    'defectetedAstma_weed', 'dieback', 'disease in leaf', 'disease inamla',
    'doddpathre', 'drumstick', 'ekka', 'eucalpytus', 'fasiation', 'ganigale',
    'ganike', 'gasagase', 'ginger', 'globe amarnath', 'henna', 'honge',
    'insulin', 'kamakasturi', 'kasamburga', 'late blightcaricature',
    'leaf blotch', 'leaf curl', 'leaf rust', 'leaf spot', 'lemon',
    'lemongrass', 'malabar_nut', 'mango', 'marigold', 'mint', 'neem',
    'nerale', 'nooni', 'onion', 'padri', 'papaya', 'pea', 'pepper',
    'pomgrana', 'pomgranate', 'powdery mildew', 'pumpkin', 'raddish', 'rose',
    'sampige', 'sapota', 'seedling blight', 'seethaashoka', 'seethaphala',
    'sooty mould', 'spinach', 'tamarind', 'taro', 'tecoma', 'thambe',
    'tomato', 'tulsi', 'turmeric', 'wilt']

# Preprocessing
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



# Load model
class YolactONNX:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

    def infer(self, image):
        input_tensor = preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        return outputs


if __name__ == "__main__":
    model_path = "yolact_base_1666_800000_.onnx"
    image_path = "test_images/alovera_1.jpg"

    yolact = YolactONNX(model_path)
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]
    
    outputs = yolact.infer(image)
    masks, classes, scores, boxes = postprocess(outputs, (orig_h, orig_w))

    for i, mask in enumerate(masks):
        color = np.random.randint(0, 255, 3).tolist()
        image[mask] = image[mask] * 0.5 + np.array(color, dtype=np.float32) * 0.5

        x1, y1, x2, y2 = boxes[i]
        x1, y1, x2, y2 = map(int, [x1 * orig_w, y1 * orig_h, x2 * orig_w, y2 * orig_h])
        label = f"{class_names[int(classes[i])]} {scores[i]:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1+10, max(y1 - 5, 0)+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Result", image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
