#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
import itertools
import json
from numpy.core.fromnumeric import argmax

#SECTION 1: UNDISTORT FISHEYE

#Read in OpenCV compatible instrinsics & distortion coeffs
COLOR_INTRINSIC = np.load('./savedCoeff/colorIntr.npy')
COLOR_DIST = np.load('./savedCoeff/colorDist.npy')
IR_INTRINSIC = np.load('./savedCoeff/irIntr.npy')
IR_DIST = np.load('./savedCoeff/irDist.npy')

print('Undistorting images-----------------')
imageDir = 'december_callibration_images'
ir_images = glob.glob('./' + imageDir + '/ir-*.png')
DIMS = (1024, 1024)
IDENTITY = np.eye(3)
for i in range(len(ir_images)):
	ir_img = cv2.imread(ir_images[i], cv2.IMREAD_UNCHANGED)
	new_K, roi = cv2.getOptimalNewCameraMatrix(IR_INTRINSIC, IR_DIST, DIMS, 1)
	map1, map2 = cv2.initUndistortRectifyMap(IR_INTRINSIC, IR_DIST, IDENTITY, new_K, DIMS, cv2.CV_32FC1)
	undistorted_ir_img = cv2.remap(ir_img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
	#save the undistorted image
	cv2.imwrite('./undistorted_december_ir_images/' + 'ir-' + str(i) + '.png', undistorted_ir_img)
