import os
import numpy as np
import cv2

data_list_file = "./data_paths/hammer_test.txt"
output_path = "./experiments/PPFT_test_2024-03-15-11:24:47/d-tof/visualization"

data_list = open(data_list_file, 'r').read().split("\n")
data_list = [s for s in data_list if (len(s) > 3)]
output_list = {}
for sample_dir in os.listdir(output_path):
    if not "sample-" in sample_dir:
        continue
    
    full_sample_dir = os.path.join(output_path, sample_dir)
    output_list[sample_dir.replace("sample-", "")] = full_sample_dir

scenes = [s.split("/")[1] for s in data_list]
scenes = set(scenes)

print("==> Following scenes are present in the test set:")
for s in scenes:
    print("    {}.".format(s))

for s in scenes:
    print("==> Processing scene {}.".format(s))
    
    files = [f for f in data_list if s in f]
    
    print("    ==> We have {} samples in this scene.".format(len(files)))
    
    # -- get file paths --
    rgbs = [f.replace("DATA_ROOT", "./data/hammer_polar") for f in files]
    deps = []
    gts = []
    outs = []
    for i, rgb in enumerate(rgbs):
        outs.append(os.path.join(output_list[str(i)], "out.png"))
        deps.append(os.path.join(output_list[str(i)], "sparse.png"))
        gts.append(os.path.join(output_list[str(i)], "gt.png"))
    
    # -- get list of images --
    rgb_list = [cv2.imread(rgb, -1) for rgb in rgbs]
    dep_list = [cv2.imread(dep, -1) for dep in deps]
    gt_list = [cv2.imread(gt, -1) for gt in gts]
    out_list = [cv2.imread(out, -1) for out in outs]
    
    # -- make videos --
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    os.makedirs("./self/videos/{}".format(s), exist_ok=True)
    
    # -- rgb --
    video=cv2.VideoWriter('./self/videos/{}/rgb.avi'.format(s), fourcc, 17, (rgb_list[0].shape[1], rgb_list[0].shape[0]))
    for i in range(len(rgb_list)):
        video.write(rgb_list[i])
    video.release()
    # -- dep --
    video=cv2.VideoWriter('./self/videos/{}/dep.avi'.format(s), fourcc, 17, (dep_list[0].shape[1], dep_list[0].shape[0]))
    for i in range(len(dep_list)):
        video.write(dep_list[i])
    video.release()
    # -- gt --
    video=cv2.VideoWriter('./self/videos/{}/gt.avi'.format(s), fourcc, 17, (gt_list[0].shape[1], gt_list[0].shape[0]))
    for i in range(len(gt_list)):
        video.write(gt_list[i])
    video.release()
    # -- out --
    video=cv2.VideoWriter('./self/videos/{}/out.avi'.format(s), fourcc, 17, (out_list[0].shape[1], out_list[0].shape[0]))
    for i in range(len(out_list)):
        video.write(out_list[i])
    video.release()