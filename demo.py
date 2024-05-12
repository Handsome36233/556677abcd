import os
import cv2
import onnxruntime

import detect_utils
from detect_utils import detect, padding_box

# yolo 配置
detect_utils.OBJ_THRESH = 0.6
detect_utils.NMS_THRESH = 0.25
size = (544, 960)
detect_onnx = "./yolov5s.onnx"
anchors = [[26, 26], [56, 16], [116, 19], [235, 23], [540, 32],
           [139, 139], [413, 64], [364, 292], [479, 703]]
detect_session = onnxruntime.InferenceSession(detect_onnx)

# demo
className2label = {"Text": 0, "Image": 1, "PageIndicator": 2, "UpperTaskBar": 3, "TextButton": 4,
                   "EditText": 5, "Icon": 6, "BackgroundImage": 7, "Drawer": 8, "Toolbar": 9, "Modal": 10,
                   "Others": 14, "Card": 11, "CheckedTextView": 12, "Switch": 13
                   }
label2className = dict(zip(className2label.values(), className2label.keys()))
image = cv2.imread("./test_image/108.jpg")
H, W, _ = image.shape
boxes, labels, scores = detect(image, detect_session, anchors, size=size)
for box, l in zip(boxes, labels):
    # xmin, ymin, xmax, ymax = padding_box(box)    
    xmin, ymin, xmax, ymax = box
    xmin = int(xmin * W / size[0])
    ymin = int(ymin * H / size[1])
    xmax = int(xmax * W / size[0])
    ymax = int(ymax * H / size[1])
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    # cv2.putText(image, label2className[l], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

image = cv2.resize(image, (W // 2, H // 2))
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
