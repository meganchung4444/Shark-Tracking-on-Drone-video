# to download in json file
import json
import numpy as np


json_path = "./assets/example_vid_6_trimmed_and_zoomed_sample_data.json"

with open(json_path, "r") as file:
  data = json.load(file)
# retrieve the x and y coordinates along with the time stampe
# and graph it

def get_box_center(box):
    x1 = box["x1"]
    x2 = box["x2"]
    y1 = box["y1"]
    y2 = box["y2"]
    return (int(x1 + ((x2 - x1)//2)), int(y1 + ((y2 - y1)//2)))

time = np.array([])
x_coor = np.array([])
y_coor = np.array([])

range1 = range(140, 800)
for frame in data:
  frame_cnt = frame.get("frame_cnt", None)
  if frame_cnt is not None and frame_cnt in range1:
    shark = frame["shark"]
    if shark is not None:
      x, y = get_box_center(shark["box"])
      x_coor = np.append(x_coor, x)
      y_coor = np.append(y_coor, y)
      time = np.append(time, frame_cnt)
coordinates = np.column_stack((x_coor, y_coor))
np.save('coordinates.npy', coordinates)
# print("\nx coordinates:", x_coor)
# print("y coordinates:", y_coor)
# print("time coordinates:", time)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 6))
# plt.plot(x_coor, y_coor, marker='o', linestyle='-', color='b', label='Measured Coordinates')
# plt.title('Shark Trajectory Detected')
# plt.xlabel('X Coordinates')
# plt.ylabel('Y Coordinates')
# plt.legend()
# plt.grid(True)
# plt.xlim(0, max(x_coor) + 100)
# plt.ylim(0, max(y_coor) + 100)

# plt.show()