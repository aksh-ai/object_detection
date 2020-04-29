import cv2
import torch
import random
import numpy as np
import PIL.Image as Image
from torchvision import transforms, models

device_name = "cuda:0:" if torch.cuda.is_available() else "cpu"
# device_name = "cpu"
device = torch.device(device_name)

model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)

model.to(device)

model.eval()

Transform = transforms.Compose([transforms.ToTensor()])

LABELS = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def color_masks(img):
    colors = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180], [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    
    r = np.zeros_like(img).astype(np.uint8)
    g = np.zeros_like(img).astype(np.uint8)
    b = np.zeros_like(img).astype(np.uint8)

    r[img == 1], g[img == 1], b[img == 1] = colors[random.randrange(0, 10)]

    color = np.stack([r, g, b], axis=2)

    return color

def predict(img_path, threshold):
    img = Image.open(img_path)
    img = Transform(img)

    if device_name=="cuda:0:":
        img =img.cuda()

        predictions = model([img])

        pred_label = [LABELS[i] for i in list(predictions[0]['labels'].cpu().numpy())]
        pred_bbox = [[(box[0], box[1]), (box[2], box[3])] for box in list(predictions[0]['boxes'].cpu().detach().numpy())]
        pred_masks = (predictions[0]['masks']>0.5).squeeze().detach().cpu().numpy()
        confidence_score = list(predictions[0]['scores'].cpu().detach().numpy())

    else:
        predictions = model([img])

        pred_label = [LABELS[i] for i in list(predictions[0]['labels'].numpy())]
        pred_bbox = [[(box[0], box[1]), (box[2], box[3])] for box in list(predictions[0]['boxes'].detach().numpy())]
        pred_masks = (predictions[0]['masks']>0.5).squeeze().detach().numpy()
        confidence_score = list(predictions[0]['scores'].detach().numpy())

    thresholded = [confidence_score.index(i) for i in confidence_score if i > threshold][-1]

    return pred_bbox[:thresholded+1], pred_label[:thresholded+1], pred_masks[:thresholded+1], confidence_score[:thresholded+1]   

def run_detection(img_path, threshold=0.6, box_thickness=3, font_size=3, font_thickness=3):
    bboxes, classes, masks, scores = predict(img_path, threshold)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    for i in range(len(bboxes)):
        text = "Label: {} | Confidence: {:.2f}".format(classes[i], scores[i])
        img = cv2.addWeighted(img, 1, color_masks(masks[i]), 0.5, 0)
        cv2.rectangle(img, bboxes[i][0], bboxes[i][1], color=(0, 255, 0), thickness=box_thickness)
        cv2.putText(img, text, bboxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255,0), thickness=font_thickness)

    img = cv2.resize(img, (1280, 720))
    
    cv2.imshow('Segmentation', img)

    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
        exit()

image = input('Enter image path: ')

run_detection(image, threshold=0.8, font_size=3)