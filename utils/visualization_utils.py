import cv2
import numpy as np
import torch
import os

def depth_to_colormap(depth, max_depth):
    """
    depth: (torch.Tensor) of shape (1, H, W)
    max_depth: the maximum depth used to normalize depth values into 0-255 for visualization
    """
    npy_depth = depth.detach().cpu().numpy()[0]
    vis = ((npy_depth / max_depth) * 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    return vis

def norm_to_colormap(norm):
    """
    norm: (torch.Tensor) of shape (3, H, W)
    """
    norm = norm.permute(1,2,0).detach().cpu().numpy()
    vis = ((norm+1) * 255/2).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    return vis

def save_visualization(pred, gt, dep, folder):
    out = depth_to_colormap(pred, 2.6)
    gt = depth_to_colormap(gt, 2.6)
    sparse = depth_to_colormap(dep, 2.6)

    cv2.imwrite(os.path.join(folder, "out.png"), out)
    cv2.imwrite(os.path.join(folder, "sparse.png"), sparse)
    cv2.imwrite(os.path.join(folder, "gt.png"), gt)