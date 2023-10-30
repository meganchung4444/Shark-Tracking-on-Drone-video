from scipy.interpolate import LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt

"""
    frame_tracker = []
    d = np.loadtxt("sample.txt")
    x = d.T[0]
    y = d.T[1]
    frames = np.arange(len(x))
"""


d = np.loadtxt("sample.txt")
frame_cnt = 100 # example
x = d.T[0]
y = d.T[1]
z = np.hypot(x, y)
X = np.linspace(min(x), max(x), num=frame_cnt)
Y = np.linspace(min(y), max(y), num=frame_cnt)
X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
interp = LinearNDInterpolator(list(zip(x, y)), z)
Z = interp(X, Y)
print(X, Y)

for i in range(frame_cnt):
    x_i = np.average(np.nansum(Z[i]))
    y_i = np.average(np.nansum(Z.T[i]))

# plt.pcolormesh(X, Y, Z, shading='auto')
# plt.plot(x, y, "ok", label="input point")
# plt.legend()
# plt.colorbar()
# plt.axis("equal")
# plt.show()