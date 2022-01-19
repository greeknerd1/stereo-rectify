#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
import itertools
import json
from numpy.core.fromnumeric import argmax



#CHECKERBOARD CALLIBRATION ON UNDISTORTED IR

# Defining the dimensions of checkerboard
CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints_color = [] 
imgpoints_ir = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None


#Set color/ir images directory
imageDir = 'raw_december_callibration_images'

for camType in ['color', 'ir']:
    images = glob.glob('./' + imageDir + '/' + camType + '*.png')
    for fname in images:
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

        if camType == 'color':
            gray = img
        else: # Convert IR from 16 bit gray scale to 8 bit gray scale
            gray = (img / 256).astype(np.uint8)

        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 0)

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            if camType == 'color': #only needs to create obj coordinates once
                objpoints.append(objp)
            # refining pixel coordinates for given 2d points.

            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            
            if camType == 'color':
                imgpoints_color.append(corners2)
            else: # IR camera
                imgpoints_ir.append(corners2)

            # Draw and display the corners
            gray = cv2.drawChessboardCorners(gray, CHECKERBOARD, corners2, ret)
        else:
            print('Delete:', fname)
        
        # cv2.imshow('img', gray)
        # cv2.waitKey(0)

    cv2.destroyAllWindows()

    h,w = gray.shape[:2]
    print('type:', camType, 'height:', h, 'width:', w)
    
    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    if camType == 'color':
        ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints_color, (w, h), None, None)
        print('Color camera callibration rms:', ret1)
    else: #IR camera callibration
        ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints_ir, (w, h), None, None)
        print('IR camera callibration rms:', ret2)

print('color callibrated instrinsic', mtx1)
print('color callibrated distortion', dist1)
print('ir callibrated instrinsic', mtx2)
print('ir callibrated distortion', dist2)

# all_flags = [cv2.CALIB_FIX_INTRINSIC, cv2.CALIB_USE_INTRINSIC_GUESS, cv2.CALIB_FIX_PRINCIPAL_POINT, \
#                                 cv2.CALIB_FIX_FOCAL_LENGTH, cv2.CALIB_FIX_ASPECT_RATIO, cv2.CALIB_SAME_FOCAL_LENGTH, \
#                                 cv2.CALIB_ZERO_TANGENT_DIST, cv2.CALIB_RATIONAL_MODEL] #WITHOUT  cv2.CALIB_FIX_S1_S2_S3_S4, cv2.CALIB_TILTED_MODEL, cv2.CALIB_FIX_TAUX_TAUY

all_flags = [cv2.CALIB_FIX_INTRINSIC, cv2.CALIB_FIX_PRINCIPAL_POINT, cv2.CALIB_FIX_FOCAL_LENGTH, cv2.CALIB_FIX_K1, cv2.CALIB_FIX_K2, cv2.CALIB_FIX_K3, cv2.CALIB_FIX_K4, cv2.CALIB_FIX_K5, cv2.CALIB_FIX_K6]

#Read in OpenCV compatible instrinsics & distortion coeffs
COLOR_INTRINSIC = np.load('./savedCoeff/colorIntr.npy')
COLOR_DIST = np.load('./savedCoeff/colorDist.npy')
IR_INTRINSIC = np.load('./savedCoeff/irIntr.npy')
IR_DIST = np.load('./savedCoeff/irDist.npy')

print("FACTORY INTRINSINCS")
print("color intr")
print(COLOR_INTRINSIC)
print("color dist")
print(COLOR_DIST)
print("ir intrin")
print(IR_INTRINSIC)
print("ir dist")
print(IR_DIST)

w, h = 1280, 720

print('Calling Stereo Callibrate-------------------')
#Approach 1: Using factory instrinsics
rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_color, imgpoints_ir, COLOR_INTRINSIC, COLOR_DIST, IR_INTRINSIC, IR_DIST, (w, h), \
                                flags=0
                                + cv2.CALIB_FIX_INTRINSIC
                                + cv2.CALIB_FIX_PRINCIPAL_POINT
                                + cv2.CALIB_FIX_FOCAL_LENGTH
                                + cv2.CALIB_FIX_K1
                                + cv2.CALIB_FIX_K2
                                + cv2.CALIB_FIX_K3
                                + cv2.CALIB_FIX_K4
                                + cv2.CALIB_FIX_K5
                                + cv2.CALIB_FIX_K6)
                                #+ cv2.CALIB_USE_INTRINSIC_GUESS
                                #+ cv2.CALIB_RATIONAL_MODEL)
                                #+ cv2.CALIB_FIX_ASPECT_RATIO) 
                                #+ cv2.CALIB_ZERO_TANGENT_DIST)
print("Stereo calibration rms: ", rms)

print('Calling Stereo Rectify-------------------')
R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
x1, y1, w1, h1 = roi_left  
x2, y2, w2, h2 = roi_right
print("ROIS", roi_left, roi_right)

#Saving coefficients
leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)


print('Rectifying images-----------------')
color_images = glob.glob('./' + imageDir + '/color-*.png')
ir_images = glob.glob('./' + imageDir + '/ir-*.png')
for i in range(len(color_images)):
    leftFrame = cv2.imread(color_images[i], cv2.IMREAD_UNCHANGED)

    rightFrame = cv2.imread(ir_images[i], cv2.IMREAD_UNCHANGED)
    rightFrame = (rightFrame / 256).astype(np.uint8) #conv from 16 bit grayscale to 8 bit #recomment

    left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR)

    for y in range(0, 1200, 30): #Draw lines to validate rectification
        line_thickness = 1
        cv2.line(left_rectified, (0, y), (1600, y), (0, 255, 0), thickness=line_thickness)
        cv2.line(right_rectified, (0, y), (1600, y), (0, 255, 0), thickness=line_thickness)

    cv2.imshow("Initial Color", leftFrame)
    cv2.imshow("Initial IR", rightFrame)
    cv2.imshow("Rectified Color", left_rectified)
    cv2.imshow("Rectified IR", right_rectified)

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
