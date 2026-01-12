# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

import numpy as np

###############################################################################
# Utility functions
def _check_color_dim(val): ...

###############################################################################
# RGB<->HEX conversion

def _hex_to_rgba(hexs): ...
def _rgb_to_hex(rgbs): ...

###############################################################################
# RGB<->HSV conversion

def _rgb_to_hsv(rgbs): ...
def _hsv_to_rgb(hsvs): ...

###############################################################################
# RGB<->CIELab conversion

# These numbers are adapted from MIT-licensed MATLAB code for
# Lab<->RGB conversion. They provide an XYZ<->RGB conversion matrices,
# w/D65 white point normalization built in.

# _rgb2xyz = np.array([[0.412453, 0.357580, 0.180423],
#                     [0.212671, 0.715160, 0.072169],
#                     [0.019334, 0.119193, 0.950227]])
# _white_norm = np.array([0.950456, 1.0, 1.088754])
# _rgb2xyz /= _white_norm[:, np.newaxis]
# _rgb2xyz_norm = _rgb2xyz.T
_rgb2xyz_norm = ...

# _xyz2rgb = np.array([[3.240479, -1.537150, -0.498535],
#                     [-0.969256, 1.875992, 0.041556],
#                     [0.055648, -0.204043, 1.057311]])
# _white_norm = np.array([0.950456, 1., 1.088754])
# _xyz2rgb *= _white_norm[np.newaxis, :]
_xyz2rgb_norm = ...

def _rgb_to_lab(rgbs): ...
def _lab_to_rgb(labs): ...
