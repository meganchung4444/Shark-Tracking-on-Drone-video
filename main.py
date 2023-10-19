from collections import defaultdict
import cv2
import numpy as np
import json
from ultralytics import YOLO
import supervision as sv
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from fast_sam_ubuntu import FastSAMCustom
# from sam_ubuntu import SAMCustom

def load_video(video_path, output_path="./results/output.mp4"):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    return cap, out

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def display_frame(annotated_frame):
        
    # Display the annotated frame
    resize = ResizeWithAspectRatio(annotated_frame, width=1200, height=800)
    cv2.imshow("YOLOv8 Tracking", resize)

    # Resizing the window
    cv2.resizeWindow("YOLOv8 Tracking", 1200, 800)


def is_shark_missed(json_format):
    missed = True
    for obj in json_format:
        if obj["name"] == "shark":
            missed = False
    return missed

# Reference: https://stackoverflow.com/questions/40795709/checking-whether-two-rectangles-overlap-in-python-using-two-bottom-left-corners
def is_overlapping(rec1, rec2):
    print("Testing overlapping between", rec1, "and", rec2)
    if (rec2['x2'] > rec1['x1'] and rec2['x2'] < rec1['x2']) or (rec2['x1'] > rec1['x1'] and rec2['x1'] < rec1['x2']):
        x_match = True
    else:
        x_match = False
    if (rec2['y2'] > rec1['y1'] and rec2['y2'] < rec1['y2']) or (rec2['y1'] > rec1['y1'] and rec2['y1'] < rec1['y2']):
        y_match = True
    else:
        y_match = False
    if x_match and y_match:
        return True
    else:
        return False

def find_sharks_by_sam(prev_sharks_prediction_list, sam_results):
    results = []
    for shark in prev_sharks_prediction_list:
        shark_name, shark_cls, shark_box, shark_confidence = shark
        for obj in sam_results:
            obj_name, obj_cls, obj_box, obj_confidence = obj
            if is_overlapping(shark_box, obj_box):
                results.append(('shark', obj_cls, obj_box, obj_confidence))
    return results

def draw_tracking_history(frame, tracking_history):
    if len(tracking_history) > 1:
        for i in range(1, len(tracking_history)):
            cv2.line(frame, tracking_history[i-1], tracking_history[i], (0, 255, 0), thickness=3, lineType=8)

# 1. Set up a model
model = YOLO("best.pt")
model.to("cuda")

# 2. Video Setup
# Open the video file
video_path = "./assets/example_vid_1.mp4"
save_dir = "./results"
results_name = "test_ori"
output_path = f'{save_dir}/{results_name}.mp4'
cap, video_writer = load_video(video_path, output_path)


fs = FastSAMCustom()

# 3. SAM setup
# Set up SAM


# Store all sharks from the most recent successed frame.
prev_sharks_prediction_list = []
tracking_history = []

# Loop through the video frames
frame_cnt = 1
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model(frame)
        
        """
        # Custom SAM
        sam_results = fs.get_json_data(frame=frame)
        if sam_results:
            for idx, r in enumerate(sam_results):
                # print(f"SAM {frame_cnt}-{idx}", r)
                continue
        else:
            print("No SAM results")
            # Doc: https://docs.ultralytics.com/modes/predict/#boxes
        """

        # YOLO
        for idx, r in enumerate(results):
            # Returns torch.Tensor(https://pytorch.org/docs/stable/tensors.html)
            # xywh returns the boxes in xywh format.
            # cpu() moves the object into cpu
            boxes = r.boxes.xywh.cpu()

            # Store all predicted objects
            object_prediction_list = []
            sharks_prediction_list = []
            
            # Contains box plot information
            yolo_json_format = json.loads(r.tojson())
            # print(f"YOLO {frame_cnt}-{idx}", yolo_json_format)

            # Construct all object list and shark list
            for obj in yolo_json_format:     
                name = obj["name"]
                cls = obj["class"]
                box = obj["box"]
                confidence = obj["confidence"]
                object_prediction_list.append((name, cls, box, confidence))
                
                # Store only sharks
                if name == "shark":
                    sharks_prediction_list.append((name, cls, box, confidence))
           
            """
            # Check if shark object is missing in yolo
            if len(sharks_prediction_list) == 0 and sam_results:
                print("Sharks are missing")
                sharks_prediction_list = find_sharks_by_sam(prev_sharks_prediction_list, sam_results)
                print("Detected sharks using SAM", sharks_prediction_list)
            """

            # Plot on each frame
            for obj in object_prediction_list:
                name, cls, box, confidence = obj

                # Get Coordinates for each obj
                x1 = box["x1"]
                x2 = box["x2"]
                y1 = box["y1"]
                y2 = box["y2"]

                center = (int(x1 + ((x2 - x1)//2)), int(y1 + ((y2 - y1)//2)))
                label_pos = (center[0], center[1]-2)
                # Draw Box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (56, 56, 255), 2)
                
                # Append Tracking History
                if name == 'shark':
                    tracking_history.append(center)
                
                    # Plot current location on the frame
                    cv2.circle(frame, center, 5, (56, 56, 255), 3)
                
                # Generate Labels
                
                cv2.putText(frame,
                            "Shark", label_pos,
                            0,
                            0.6, [56, 56, 255],
                            thickness=1,
                            lineType=cv2.LINE_AA)
                


        prev_sharks_prediction_list = sharks_prediction_list
        draw_tracking_history(frame, tracking_history)
        video_writer.write(frame)
        frame_cnt+=1
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
video_writer.release()
cap.release()

