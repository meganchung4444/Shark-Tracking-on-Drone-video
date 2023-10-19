# Honor Thesis

## Abstract

## Installment

#### 1. Install all required dependancies

- Segment-Anything [link](https://github.com/facebookresearch/segment-anything)

#### 2. Download pretrained model

#### 3. Download sam_vit_h
doc.txt
Google Drive Link for drone footage: https://drive.google.com/drive/folders/1NDtOnPW13CkehLVdpn_KYgg7iTYxfFD5
Weight of the model link: https://universe.roboflow.com/shark-detection/shark-drone/model/3


#### 4. Useful commands
```
scp test.mp4 10.33.113.224:C:\\Users\\sunbe\\
```
python yolo_video_ubuntu_test.py ./assets/example_vid_3.mp4 results/test.mp4 0.6
python yolo_video_ubuntu_test.py best.pt ./assets/example_vid_3.mp4 results/test.mp4 0.6
python yolo_video_ubuntu_test.py best.pt ./assets/example_vid_3.mp4 results/test.mp4 0.6
