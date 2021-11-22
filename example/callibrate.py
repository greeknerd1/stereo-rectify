#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
import itertools
import json
from numpy.core.fromnumeric import argmax


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


color_shape = (720, 1280)
ir_shape = (1024, 1024) #Passive IR shape is 1024 x 1024
ir_crop_x = 50
ir_crop_y = 260

color_new_width = 720 / 576 * 640
color_crop = int((color_shape[1] - color_new_width) / 2)

# Extracting path of individual image stored in a given directory
imageDir = 'outside_checker_straight_ir_cropped' #equalizedImagesNearBetter (Equalized R and Equalized IR, most promising)
#Demo on:
#equalizedImagesFarBetter (Equalized R and Scaled Equalized IR, noisy)
#only_ir_scaled_image_set (R and Scaled IR, resembles SDK viewer)
#both_hist_equalized_image_set1 (Equalized R Equalized IR additional set)

for camType in ['color', 'ir']:
    images = glob.glob('./' + imageDir + '/' + camType + '*.png')
    for fname in images:
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

        if camType == 'color':
            #img = img[:,color_crop:-color_crop] #Crop width
            #img = cv2.resize(img, (640, 576)) #Resize
            #gray = img[:, :, 2] #Take R channel only #recomment
            gray = img
        else: # Convert IR from 16 bit gray scale to 8 bit gray scale
            gray = (img / 256).astype(np.uint8)
            gray = gray[ir_crop_y: ir_shape[0] - ir_crop_y, ir_crop_x:ir_shape[1]-ir_crop_x]
            gray = cv2.resize(gray, (1280, 720)) #Resize

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

            corners2 = cv2.cornerSubPix(gray, corners, (3,3),(-1,-1), criteria)
            
            if camType == 'color':
                imgpoints_color.append(corners2)
            else: # IR camera
                imgpoints_ir.append(corners2)

            # Draw and display the corners
            gray = cv2.drawChessboardCorners(gray, CHECKERBOARD, corners2, ret)
        else:
            print('Delete:', fname)
        
        cv2.imshow('img', gray)
        cv2.waitKey(0)

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

print('color callibrated instrinsic \n', mtx1)
print('color callibrated distortion \n', dist1)





all_flags = [cv2.CALIB_FIX_INTRINSIC, cv2.CALIB_USE_INTRINSIC_GUESS, cv2.CALIB_FIX_PRINCIPAL_POINT, \
                                cv2.CALIB_FIX_FOCAL_LENGTH, cv2.CALIB_FIX_ASPECT_RATIO, cv2.CALIB_SAME_FOCAL_LENGTH, \
                                cv2.CALIB_ZERO_TANGENT_DIST, cv2.CALIB_RATIONAL_MODEL] #WITHOUT  cv2.CALIB_FIX_S1_S2_S3_S4, cv2.CALIB_TILTED_MODEL, cv2.CALIB_FIX_TAUX_TAUY


# print('[cv2.CALIB_FIX_INTRINSIC (256), cv2.CALIB_USE_INTRINSIC_GUESS (1), cv2.CALIB_FIX_PRINCIPAL_POINT (4), \
#                                 cv2.CALIB_FIX_FOCAL_LENGTH (16), cv2.CALIB_FIX_ASPECT_RATIO (2), cv2.CALIB_SAME_FOCAL_LENGTH (512), \
#                                 cv2.CALIB_ZERO_TANGENT_DIST (8), cv2.CALIB_RATIONAL_MODEL (16384), \
#                                 cv2.CALIB_FIX_S1_S2_S3_S4 (65536), cv2.CALIB_TILTED_MODEL (262144), cv2.CALIB_FIX_TAUX_TAUY (524288)]')

# flagsComb = []
# for L in range(0, len(all_flags)+1):
#     for subset in itertools.combinations(all_flags, L):
#         flagsComb.append(subset)

# results = []
# i = 0
# for f in flagsComb:
#     print(str(i) + '/' + str(len(flagsComb)))
#     i += 1

#     rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_color, imgpoints_ir, mtx1, dist1, mtx2, dist2, (w, h), flags=sum(f))
#     R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
#     x1, y1, w1, h1 = roi_left  
#     x2, y2, w2, h2 = roi_right
#     area = (w1 * h1) + (w2 * h2)
#     if rms < 6 and area > 0:
#         results.append((area, sum(f), f, rms))
        

# x = sorted(results)
# for elem in x:
#     print(elem)

# with open("ir_cropped_stereo_callibrate_optimization.txt", 'w') as f:
#     for elem in x:
#         f.write(str(elem) + "\n")
#     f.close()
    

# for _ in range(10):
#     print(cv2.stereoCalibrate(objpoints, imgpoints_color, imgpoints_ir, mtx1, dist1, mtx2, dist2, (w, h), flags=525)[0])
#     print('---------')
# exit()



# print('Calling Stereo Callibrate-------------------')
# rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_color, imgpoints_ir, mtx1, dist1, mtx2, dist2, (w, h), \
#                                 flags=0
#                                 #+ cv2.CALIB_FIX_INTRINSIC)
#                                 + cv2.CALIB_USE_INTRINSIC_GUESS) #this for near, far
#                                 #+ cv2.CALIB_FIX_PRINCIPAL_POINT
#                                 #+ cv2.CALIB_FIX_FOCAL_LENGTH #this for near
#                                 #+ cv2.CALIB_FIX_ASPECT_RATIO #this for near, far
#                                 #+ cv2.CALIB_SAME_FOCAL_LENGTH #this for far
#                                 #+ cv2.CALIB_ZERO_TANGENT_DIST) #this for near, far
#                                 #+ cv2.CALIB_RATIONAL_MODEL) #this for near, far
#                                 #+ cv2.CALIB_THIN_PRISM_MODEL)
#                                 #+ cv2.CALIB_FIX_S1_S2_S3_S4 #maybe for far
#                                 #+ cv2.CALIB_TILTED_MODEL  
#                                 #+ cv2.CALIB_FIX_TAUX_TAUY) #maybe for far

# print("Stereo calibration rms: ", rms)
                                                                                                                        




# #Read in camera instrinsics and distortion coefficients
# with open("calibration_data", "r") as f:
#     calibration_raw = f.read()
# calibration_json = json.loads(calibration_raw)
# cx, cy, fx, fy, k1, k2, k3, k4, k5, k6, codx, cody, p2, p1 = calibration_json['CalibrationInformation']['Cameras'][0]['Intrinsics']['ModelParameters']
# COLOR_INTRINSIC = np.array([[fx * 1280, 0, cx * 1280], [0, fy * 720, cy * 720], [0, 0, 1]])
# COLOR_DIST = np.array([k1, k2, p1, p2, k3, k4, k5, k6])
# cx, cy, fx, fy, k1, k2, k3, k4, k5, k6, codx, cody, p2, p1 = calibration_json['CalibrationInformation']['Cameras'][1]['Intrinsics']['ModelParameters']
# IR_INTRINSIC = np.array([[fx * w, 0, cx * w], [0, fy * h, cy * h], [0, 0, 1]])
# IR_DIST = np.array([k1, k2, p1, p2, k3, k4, k5, k6])

# print("FACTORY INTRINSINCS")
# print("color intr")
# print(COLOR_INTRINSIC)
# print("color dist")
# print(COLOR_DIST)
# print("ir intrin")
# print(IR_INTRINSIC)
# print("ir dist")
# print(IR_DIST)


# print('Calling Stereo Callibrate-------------------')
# rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_color, imgpoints_ir, COLOR_INTRINSIC, COLOR_DIST, IR_INTRINSIC, IR_DIST, (w, h), \
#                                 flags=0
#                                 + cv2.CALIB_FIX_INTRINSIC
#                                 + cv2.CALIB_FIX_PRINCIPAL_POINT
#                                 + cv2.CALIB_FIX_FOCAL_LENGTH
#                                 + cv2.CALIB_FIX_K1
#                                 + cv2.CALIB_FIX_K2
#                                 + cv2.CALIB_FIX_K3
#                                 + cv2.CALIB_FIX_K4
#                                 + cv2.CALIB_FIX_K5
#                                 + cv2.CALIB_FIX_K6
#                                 + cv2.CALIB_USE_INTRINSIC_GUESS)
#                                 #+ cv2.CALIB_RATIONAL_MODEL)
#                                 #+ cv2.CALIB_FIX_ASPECT_RATIO) #this for near, far
#                                 #+ cv2.CALIB_ZERO_TANGENT_DIST) #this for near, far

# print("Stereo calibration rms: ", rms)

w, h = 1280, 720

rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_color, imgpoints_ir, mtx1, dist1, mtx2, dist2, (w, h), flags=518) #530(with4) #540 774 16927 531 16387(without ir4)
print("Stereo calibration rms: ", rms)              

print('Calling Stereo Rectify-------------------')
R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
x1, y1, w1, h1 = roi_left  
x2, y2, w2, h2 = roi_right
print("ROIS", roi_left, roi_right)

#Saving coefficients
leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)

# np.save('./savedCoeff/leftMapX.npy', leftMapX)
# np.save('./savedCoeff/leftMapY.npy', leftMapY)
# np.save('./savedCoeff/rightMapX.npy', rightMapX)
# np.save('./savedCoeff/rightMapY.npy', rightMapY)

# #Load saved coefficients
# leftMapX = np.load('./savedCoeff/leftMapX.npy')
# leftMapY = np.load('./savedCoeff/leftMapY.npy')
# rightMapX = np.load('./savedCoeff/rightMapX.npy')
# rightMapY = np.load('./savedCoeff/rightMapY.npy')




print('Rectifying images-----------------')
color_images = glob.glob('./' + imageDir + '/color-*.png')
ir_images = glob.glob('./' + imageDir + '/ir-*.png')
for i in range(len(color_images)):
    leftFrame = cv2.imread(color_images[i], cv2.IMREAD_UNCHANGED)
    #leftFrame = leftFrame[:,color_crop:-color_crop]
    #leftFrame = cv2.resize(leftFrame, (640, 576))
    #leftFrame = leftFrame[:, :, 2] #crop, resize, take R channel #recomment

    rightFrame = cv2.imread(ir_images[i], cv2.IMREAD_UNCHANGED)
    rightFrame = (rightFrame / 256).astype(np.uint8) #conv from 16 bit grayscale to 8 bit #recomment
    rightFrame = rightFrame[ir_crop_y: ir_shape[0] - ir_crop_y, ir_crop_x:ir_shape[1]-ir_crop_x]
    rightFrame = cv2.resize(rightFrame, (1280, 720)) #Resize

    #leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1) #perform rectification
    left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR)
    #rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)
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


# ################################################################# Finding best alpha

# alphas = [i / 100 for i in range(101)]
# all_left_rois = []
# all_left_areas = []
# all_right_rois = []
# all_right_areas = []
# for a in alphas:
#     R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=a)
#     x1, y1, w1, h1 = roi_left
#     x2, y2, w2, h2 = roi_right
#     all_left_rois.append(roi_left)
#     all_left_areas.append(w1 * h1)
#     all_right_rois.append(roi_right)
#     all_right_areas.append(w2 * h2)

# print('Max left alpha:', argmax(all_left_areas) / 100)
# print('Max right alpha:', argmax(all_right_areas) / 100)


# ###################################################################



# #Get original image
# h,  w = leftFrame.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K1, D1, (w,h), 1, (w,h))
# print(roi)
# # undistort
# dst = cv2.undistort(leftFrame, K1, D1, None, newcameramtx)
# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imshow('Original', leftFrame)
# cv2.imshow('Cropped Result', dst)
# key = cv2.waitKey(0)
# #so basically vals returned by callibrate camera aren't good



# left_rectified = cv2.undistort(leftFrame, K1, D1, R1)
# right_rectified = cv2.undistort(rightFrame, K2, D2, R2)

# x, y, w, h = roi_left
# if w != 0 and h != 0:
#     left_rectified = left_rectified[y:y+h, x:x+w]
#     cv2.imshow("Initial Color", leftFrame)
#     cv2.imshow("Rectified Color", left_rectified)


# x, y, w, h = roi_right
# if w != 0 and h != 0:
#     right_rectified = right_rectified[y:y+h, x:x+w]
#     cv2.imshow("Initial IR", rightFrame)
#     cv2.imshow("Rectified IR", right_rectified)

# key = cv2.waitKey(0)



# key = cv2.waitKey(0)

#left_rectified = left_rectified[top_left_y:bot_left_y, top_left_x:bot_left_x]




# x, y, w, h = roi_left
# if w != 0 and h != 0:
#     left_rectified = left_rectified[y:y+h, x:x+w]
#     cv2.imshow("Initial Color", leftFrame)
#     cv2.imshow("Rectified Color", left_rectified)


# x, y, w, h = roi_right
# if w != 0 and h != 0:
#     right_rectified = right_rectified[y:y+h, x:x+w]
#     cv2.imshow("Initial IR", rightFrame)
#     cv2.imshow("Rectified IR", right_rectified)

# key = cv2.waitKey(0)





# directory = r'C:\Users\OpenARK\Desktop\pyk4a\example\images'
# os.chdir(directory)
# cv2.imwrite('color-rectified-0.png', left_rectified)
# cv2.imwrite('ir-rectified-0.png', right_rectified)