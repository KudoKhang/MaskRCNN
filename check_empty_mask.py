import os
import cv2
import numpy as np

root = 'dataset/ibug/val/masks/'

path_mask = [name for name in os.listdir(root) if name.endswith('png')]

l_err = []
for path in path_mask:
    mask = cv2.imread(root + path)
    if len(np.unique(mask)) == 1:
        l_err.append(path)

for path in l_err:
    try:
        os.remove(root + path)
        os.remove('dataset/ibug/val/images/' + path[:-3] + 'jpg')
    except:
        raise "Error"