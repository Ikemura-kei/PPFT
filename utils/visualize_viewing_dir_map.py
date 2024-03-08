import numpy as np
import cv2

viewing_direction_map = np.load("./utils/vd.npy")
print(viewing_direction_map.min())
print(viewing_direction_map.max())
vis = ((viewing_direction_map + 1 / 2) * 255).astype(np.uint8)
# vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
print(vis.min())
print(vis.max())
print(vis.shape)
cv2.imwrite("utils/vd.png", vis)