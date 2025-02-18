import cv2
import numpy as np
import torchvision
import torch
# from yolo2.utils import plot_boxes_cv2
device = torch.device("cuda:0")
import math
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import os
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def preprocess_frame(frame, input_width=512, input_height=512):

    ''' 
    original_height, original_width = frame.shape[:2]
    max_dim = max(original_height, original_width)
    top_pad = (max_dim - original_height) // 2
    bottom_pad = max_dim - original_height - top_pad
    left_pad = (max_dim - original_width) // 2
    right_pad = max_dim - original_width - left_pad
    padded_frame = cv2.copyMakeBorder(frame, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    '''
    resized_frame = cv2.resize(frame, (input_width, input_height))

    # Convert color space if needed (e.g., from BGR to RGB)
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to the range [0, 1]
    normalized_frame = rgb_frame / 255.0

    # Convert the numpy array to a PyTorch tensor
    tensor_frame = torch.tensor(normalized_frame, dtype=torch.float32)

    # PyTorch expects the channel dimension to be the first dimension, so transpose the tensor
    tensor_frame = tensor_frame.permute(2, 0, 1)

    # Expand dimensions to create a batch of size 1 (if needed)
    # This depends on the input shape expected by the RCNN model
    tensor_frame = tensor_frame.unsqueeze(0)

    return tensor_frame


def plot_boxes_cv2_rcnn(conf, img, boxes, savename=None, class_names=None, color=None):

    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]]);
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1,x2, y2, cls_conf, cls_id = box
        # if cls_conf < conf or cls_id != 1:
        #     continue
        if cls_conf < conf or cls_id >80:
            continue
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))
        
        # x1 = int(round((box[0] - box[2]/2.0)))
        # y1 = int(round((box[1] - box[3]/2.0)))
        # x2 = int(round((box[0] + box[2]/2.0)))
        # y2 = int(round((box[1] + box[3]/2.0)))

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
            
        # print('%s: %f' % (class_names[cls_id], cls_conf))
        classes = len(class_names)
        # offset = cls_id * 123457 % classes
        # red   = get_color(2, offset, classes)
        # green = get_color(1, offset, classes)
        # blue  = get_color(0, offset, classes)
        # if color is None:
        #     rgb = (red, green, blue)
        # img = cv2.putText(img, class_names[cls_id], (y1,x1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        # img = cv2.rectangle(img, (y1, x1), (y2,x2), rgb, 1)
        color = (255, 0, 255)  # Magenta color
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Put the label with a background
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(class_names[cls_id]+str(cls_conf), font, 0.5, 1)[0]
        text_x = x1
        text_y = y1
        cv2.rectangle(img, (text_x, text_y - text_size[1] - 2), (text_x + text_size[0], text_y + 2), color, -1)
        cv2.putText(img, class_names[cls_id],(text_x, text_y), font, 0.5, (0,0,0), 1, cv2.LINE_AA)

        
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


def detect_human(conf, video_path = 'input_video.mp4', output_video_path = 'output_video_with_boxes.mp4'):
    # Load the video

    cap = cv2.VideoCapture(video_path)

    # Initialize RCNN model (You'll need to replace this with actual code for model initialization)
    rcnn_model =  torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)

    # Variables for success rate calculation
    total_frames = 0
    detected_frames = 0

    # Initialize video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (512, 512))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    from tqdm import tqdm
    for frame_idx in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx <100 or frame_idx>650:
            continue

        total_frames += 1
        
        # Preprocess frame for RCNN model
        height, width, _ = frame.shape

        # If the height is greater than 512, crop it
        if height > 512:
            frame = cv2.resize(frame, (int(width * 512 / height), 512))
            # Calculate padding
            left = (512 - frame.shape[1]) // 2
            right = 512 - frame.shape[1] - left
            frame = cv2.copyMakeBorder(frame, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
        # print(frame.shape)
        resized_frame = cv2.resize(frame, (512, 512))
        processed_frame = preprocess_frame(resized_frame).to(device)

        # Detect persons using RCNN model
        try:
            output = rcnn_model(processed_frame)[0]
        except:
            continue
        boxes = output["boxes"]
        labels = output["labels"]
        scores = output["scores"]
        person = 2
        if person == 1:
            max_index = torch.argmax(scores)
            
            if labels[max_index] == 1:
                box = boxes[max_index].detach().cpu().numpy().tolist()
                label = labels[max_index].item()
                score = scores[max_index].item()
                box.append(score)
                box.append(label)
                resized_frame = plot_boxes_cv2_rcnn(conf, resized_frame, [box], class_names=class_names)
                detected_frames += 1
                
        else:
            box = boxes.detach().cpu().numpy().tolist()
            label = labels.detach().cpu().numpy().tolist()
            score = scores.detach().cpu().numpy().tolist()
            for i in range(len(box)):
                box[i].append(score[i])
                box[i].append(label[i])
            resized_frame = plot_boxes_cv2_rcnn(conf, resized_frame, box, class_names=class_names)
            detected_frames += 1

            # Write frame with bounding boxes to the output video
        # resized_frame = cv2.resize(resized_frame, (frame_width, frame_height))
        resized_frame = cv2.resize(resized_frame, (512, 512))
        cv2.imwrite("video_output/"+str(total_frames) +".png", resized_frame)
        out.write(resized_frame)

    # Release video capture and writer objects
    cap.release()
    out.release()

    # Calculate success rate
    success_rate = (detected_frames / total_frames) * 100
    print("Success Rate:", success_rate, "%")
    
def detect_human_yolov8( conf_thresh = 0.5, video_path = 'input_video.mp4', output_video_path = 'output_video_with_boxes.mp4'):
    # Load the video
    
    yolov8_model = YOLO('yolov8n.pt', task="detect")
    cap = cv2.VideoCapture(video_path)

    # Variables for success rate calculation
    total_frames = 0
    detected_frames = 0

    # Initialize video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   
    img_size = 512
    from tqdm import tqdm
    for frame_idx in tqdm(range(0, 100)):
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        
        # Preprocess frame for RCNN model
        resized_frame = cv2.resize(frame, (512, 512))
        processed_frame = preprocess_frame(resized_frame).to(device)

        # Detect persons using RCNN model
        try:
            output = yolov8_model(processed_frame)
        except:
            continue
        all_boxes = []
        scores = []
        labels = []
        flag = False
        for detection in output: #batch
            boxes = []
            detections = detection.boxes
            for idx in range(detections.xyxy.shape[0]):
                if detections.conf[idx] >= conf_thresh:
                    x1, y1, x2, y2 = map(int, detections.xyxy[idx].tolist())
                    boxes.append([x1, y1,x2,y2, detections.conf[idx].item(), int(detections.cls[idx].item())+1])
                    if int(detections.cls[idx].item()) == 0 and detections.conf[idx] >= conf_thresh:
                        flag = True
        if flag:
            detected_frames += 1   
        resized_frame = plot_boxes_cv2_rcnn(resized_frame, boxes, class_names=class_names)
        resized_frame = cv2.resize(resized_frame, (frame_width, frame_height))
        cv2.imwrite("video_output/"+str(total_frames) +".png", resized_frame)
        out.write(resized_frame)

    # Release video capture and writer objects
    cap.release()
    out.release()

    # Calculate success rate
    attack_success_rate = (1 - (detected_frames / total_frames) )* 100
    print("conf_thresh: ", conf_thresh, "Attack Success Rate:", attack_success_rate, "%")
    return attack_success_rate
    

def detect_human_yolov8_test_dataset( conf_thresh = 0.5, test_dataset_path = "/home/yjli/AIGC/Adversarial_camou/my_dataset/test_dataset/", output_video_path = 'output_video_with_boxes.mp4'):
    # Load the video
    
    yolov8_model = YOLO('yolov8n.pt', task="detect", verbose=False)

    # Variables for success rate calculation
    total_frames = 0
    detected_frames = 0

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 2, (512, 512))
    
    img_size = 512

    filelist = os.listdir(test_dataset_path)
    for frame_idx in tqdm(range(0, len(filelist))):
        # ret, frame = cap.read()
        frame = cv2.imread(test_dataset_path + filelist[frame_idx])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #if not ret:
        #    break

        total_frames += 1
        
        # Preprocess frame for RCNN model
        resized_frame = cv2.resize(frame, (512, 512))
        processed_frame = preprocess_frame(resized_frame).to(device)

        # Detect persons using RCNN model
        try:
            output = yolov8_model(processed_frame)
        except:
            continue
        all_boxes = []
        scores = []
        labels = []
        count = 0
        flag = False
        for detection in output: #batch
            boxes = []
            detections = detection.boxes
            for idx in range(detections.xyxy.shape[0]):
                if detections.conf[idx] >= conf_thresh:
                    x1, y1, x2, y2 = map(int, detections.xyxy[idx].tolist())
                    boxes.append([x1, y1,x2,y2, detections.conf[idx].item(), int(detections.cls[idx].item())+1])
                    if int(detections.cls[idx].item()) == 0 and detections.conf[idx] >= conf_thresh:
                        flag = True
        if flag:
            detected_frames += 1            
        resized_frame = plot_boxes_cv2_rcnn(resized_frame, boxes, class_names=class_names)
        cv2.imwrite("video_output/"+str(total_frames) +".png", resized_frame[...,[2,1,0]])
        out.write(resized_frame[...,[2,1,0]])

    # Release video capture and writer objects
    out.release()

    # Calculate success rate
    attack_success_rate = (1 - (detected_frames / total_frames) )* 100
    print("conf_thresh: ", conf_thresh, "Attack Success Rate:", attack_success_rate, "%")
    return attack_success_rate
    
if __name__ == "__main__":
    # detect_human(video_path = './twopeople_rcnn.mp4', output_video_path = 'twopeople_rcnn_with_boxes.mp4')
    detect_human(conf=0.7, video_path = './attack2.mp4', output_video_path = 'attack2_rcnn_with_boxes.mp4')
    """    ASR = []
        for conf_thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
            asr = detect_human_yolov8_test_dataset(conf_thresh)
            ASR.append(asr)
        conf_thresh = [0.1, 0.3, 0.5, 0.7, 0.9]
        for i in range(5):
            print("conf_thresh: ", conf_thresh[i], "Attack Success Rate:", ASR[i], "%")
            
    """
    # ASR = []
    # for conf_thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
    #     asr = detect_human_yolov8(conf_thresh, video_path = 'attack.mp4', output_video_path = 'output_video_with_boxes_ylov8.mp4')
    #     ASR.append(asr)
    # conf_thresh = [0.1, 0.3, 0.5, 0.7, 0.9]
    # for i in range(5):
    #     print("conf_thresh: ", conf_thresh[i], "Attack Success Rate:", ASR[i], "%")
    
    
    
        