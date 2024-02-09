import filecmp
import matplotlib.pyplot as plt
import numpy as np
import json

from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data

def get_box_center(box):
    x1 = box["x1"]
    x2 = box["x2"]
    y1 = box["y1"]
    y2 = box["y2"]
    return (int(x1 + ((x2 - x1)//2)), int(y1 + ((y2 - y1)//2)))

# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.5))

# =============
# First subplot
# =============
# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')

# plot a 3D surface like in the example mplot3d/surface3d_demo
f = open("sample.json")
d = json.load(f)

data_ori = []
data_x = []
data_y = []
for obj in d:
    shark = obj["shark"]
    data_ori.append(shark)
    if shark:
        x1, y1 = get_box_center(shark["box"])
        data_x.append(x1)
        data_y.append(y1)

frame_cnt = len(data_x)

data_x = np.array(data_x)
data_y = np.array(data_y)

X = np.linspace(min(data_x), max(data_x), num=frame_cnt)
Y = np.linspace(min(data_y), max(data_y), num=frame_cnt)

X, Y = np.meshgrid(X, Y)

Z = np.arange(0, frame_cnt, 1)
ax.scatter3D(X, Y, Z, c=Z, cmap='BuGn')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=10)

# ==============
# Second subplot
# ==============
# set up the axes for the second plot


plt.show()

