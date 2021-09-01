import time
import onnxruntime
import numpy as np
from PIL import Image

session = onnxruntime.InferenceSession("tiny-yolov3.onnx")
inname = [input.name for input in session.get_inputs()]
outname = [output.name for output in session.get_outputs()]

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data

def get_prediction(image_data, image_size):
    input = {
        inname[0]: image_data,
        inname[1]: image_size
    }
    t0 = time.time()
    boxes, scores, indices = session.run(outname, input)
    print("Predict Time: %ss" % (time.time() - t0))
    out_boxes, out_scores, out_classes = [], [], []
    for idx_ in indices[0]:
        out_classes.append(idx_[1])
        out_scores.append(scores[tuple(idx_)])
        idx_1 = (idx_[0], idx_[2])
        out_boxes.append(boxes[idx_1])
    return out_boxes, out_scores, out_classes

label =["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",
    "knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
    "pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor",
    "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

image = Image.open("kite.jpg")
image_data = preprocess(image)
image_size = np.array([image.size[1], image.size[0]], dtype=np.float32).reshape(1, 2)
out_boxes, out_scores, out_classes = get_prediction(image_data, image_size)
out_boxes = np.array(out_boxes).tolist()
out_scores = np.array(out_scores).tolist()
out_classes = np.array(out_classes).tolist()

for i, c in reversed(list(enumerate(out_classes))):
    print("box:", out_boxes[i])
    print("score:", out_scores[i],",", label[c])