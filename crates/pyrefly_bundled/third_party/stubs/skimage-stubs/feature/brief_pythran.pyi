import numpy as np

# pythran export _brief_loop(float32[:,:] or float64[:,:], uint8[:,:], int64[:,2], int32[:,2], int32[:,2])
def _brief_loop(image, descriptors, keypoints, pos0, pos1): ...
