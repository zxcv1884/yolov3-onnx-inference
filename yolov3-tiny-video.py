import time
import cv2
import onnxruntime
import numpy as np

session = onnxruntime.InferenceSession("tiny-yolov3.onnx")
inname = [input.name for input in session.get_inputs()]
outname = [output.name for output in session.get_outputs()]

def frame_process(frame, input_shape=(416, 416)):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_shape)
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    return image

def get_prediction(image_data, image_size):
    input = {
        inname[0]: image_data,
        inname[1]: image_size
    }
    t0 = time.time()
    boxes, scores, indices = session.run(outname, input)
    predict_time = time.time() - t0
    print("Predict Time: %ss" % (predict_time))
    out_boxes, out_scores, out_classes = [], [], []
    for idx_ in indices[0]:
        out_classes.append(idx_[1])
        out_scores.append(scores[tuple(idx_)])
        idx_1 = (idx_[0], idx_[2])
        out_boxes.append(boxes[idx_1])
    return out_boxes, out_scores, out_classes, predict_time

label =["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",
    "knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
    "pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor",
    "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

cap = cv2.VideoCapture('road.mp4')
sum_time = 0
sum_frame = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        image_data = frame_process(frame, input_shape=(416, 416))
        image_size = np.array([416, 416], dtype=np.float32).reshape(1, 2)
        out_boxes, out_scores, out_classes, predict_time = get_prediction(image_data, image_size)
        sum_time += predict_time
        sum_frame += 1
        out_boxes = np.array(out_boxes).tolist()
        out_scores = np.array(out_scores).tolist()
        out_classes = np.array(out_classes).tolist()

        for i, c in reversed(list(enumerate(out_classes))):
            print("box:", out_boxes[i])
            print("score:", out_scores[i],",", label[c])
        print("\n")

    else:
        print("-------------------------------------------------")
        print("Average Predict Time: %ss" % (sum_time / sum_frame))
        print("-------------------------------------------------\n")
        break

cap.release()
cv2.destroyAllWindows()