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


#### 5. Notes

- Interpolation Ideas:
    - Put 'nan' into x and y if nothing detected in the current frame
    - Length of x and y will be the same as frame_cnt
    - Use `np.linespace` to fill the gap between points within x and y. (new X, Y will be created)
    - Use 2D interpolation and create a matrix Z
    - Loop through Z. Z will have exactly the same length as X and Y.
        - For each loop, get the average of the sum of each row and column of Z.
        - The avg of a row will be x_i, and the avg of a column will be y_i
    - Plot the dots on the screen or video.