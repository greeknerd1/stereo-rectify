#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
import itertools
import json
from numpy.core.fromnumeric import argmax

print('Zooming in on images-----------------')
imageDir = 'raw_december_callibration_images'
color_images = glob.glob('./' + imageDir + '/color-*.png')
for i in range(len(color_images)):
	color_img = cv2.imread(color_images[i], cv2.IMREAD_UNCHANGED)
	
	#Cropped Image
	#cropped_ir_img = color_img[375:575,250:650]

	#Upscale Image
	#upscaled_ir_img = cv2.resize(cropped_ir_img, None, fx= 3.6, fy= 3.6, interpolation= cv2.INTER_LINEAR)

	#Downscale Image
	downscaled_color_img = cv2.resize(color_img, None, fx= .3125, fy= .3125, interpolation= cv2.INTER_LINEAR)

	# Displaying all the images
	#cv2.imshow("Original", color_img)
	#cv2.imshow("Cropped Image", cropped_ir_img)
	#cv2.imshow("Downscaled", downscaled_color_img)

	#save the cropped image and upscaled image
	# cv2.imwrite('./cropped_december_ir_images/' + 'ir-' + str(i) + '.png', cropped_ir_img)
	#cv2.imwrite('./upscaled_december_ir_images/' + 'ir-' + str(i) + '.png', upscaled_ir_img)
	cv2.imwrite('./downscaled_december_color_images/' + 'color-' + str(i) + '.png', downscaled_color_img)


