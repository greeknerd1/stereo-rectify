#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob

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
ir_shape = (576, 640)

color_new_width = 720 / 576 * 640
color_crop = int((color_shape[1] - color_new_width) / 2)

# Extracting path of individual image stored in a given directory
imageDir = 'equalizedImagesNearBetter'

for camType in ['color', 'ir']:
    images = glob.glob('./' + imageDir + '/' + camType + '*.png')
    for fname in images:
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

        if camType == 'color':
            img = img[:,color_crop:-color_crop] #Crop width
            img = cv2.resize(img, (640, 576)) #Resize
            #gray = img[:, :, 2] #Take R channel only #recomment
            gray = img
        else: # Convert IR from 16 bit gray scale to 8 bit gray scale
            gray = img 
            #gray = gray / 256 #recomment
            #gray = gray.astype(np.uint8) #recomment

        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

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
            img = cv2.drawChessboardCorners(gray, CHECKERBOARD, corners2, ret)
        else:
            print('Delete:', fname)
        
        cv2.imshow('img',img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    h,w = img.shape[:2]
    
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


#look at previous flags in github and try those, try not moving the camera so much
print('Calling Stereo Callibrate-------------------')
rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_color, imgpoints_ir, mtx1, dist1, mtx2, dist2, (w, h), \
                                flags=0
                                #+ cv2.CALIB_FIX_INTRINSIC
                                + cv2.CALIB_USE_INTRINSIC_GUESS #this for near, far
                                #+ cv2.CALIB_FIX_PRINCIPAL_POINT
                                + cv2.CALIB_FIX_FOCAL_LENGTH #this for near
                                + cv2.CALIB_FIX_ASPECT_RATIO #this for near, far
                                #+ cv2.CALIB_SAME_FOCAL_LENGTH #this for far
                                + cv2.CALIB_ZERO_TANGENT_DIST #this for near, far
                                + cv2.CALIB_RATIONAL_MODEL) #this for near, far
                                #+ cv2.CALIB_THIN_PRISM_MODEL)
                                #+ cv2.CALIB_FIX_S1_S2_S3_S4 #maybe for far
                                #+ cv2.CALIB_TILTED_MODEL  
                                #+ cv2.CALIB_FIX_TAUX_TAUY) #maybe for far
print("Stereo calibration rms: ", rms)


print('Calling Stereo Rectify-------------------')
R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)


#Saving coefficients
leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)

np.save('./savedCoeff/leftMapX.npy', leftMapX)
np.save('./savedCoeff/leftMapY.npy', leftMapY)
np.save('./savedCoeff/rightMapX.npy', rightMapX)
np.save('./savedCoeff/rightMapY.npy', rightMapY)

# leftMapX = np.load('./savedCoeff/leftMapX.npy')
# leftMapY = np.load('./savedCoeff/leftMapY.npy')
# rightMapX = np.load('./savedCoeff/rightMapX.npy')
# rightMapY = np.load('./savedCoeff/rightMapY.npy')




print('Rectifying images-----------------')
color_images = glob.glob('./' + imageDir + '/color-*.png')
ir_images = glob.glob('./' + imageDir + '/ir-*.png')
for i in range(len(color_images)):
    leftFrame = cv2.imread(color_images[i], cv2.IMREAD_UNCHANGED)
    leftFrame = leftFrame[:,color_crop:-color_crop]
    leftFrame = cv2.resize(leftFrame, (640, 576))
    #leftFrame = leftFrame[:, :, 2] #crop, resize, take R channel #recomment

    rightFrame = cv2.imread(ir_images[i], cv2.IMREAD_UNCHANGED)
    #rightFrame = (rightFrame / 256).astype(np.uint8) #conv from 16 bit grayscale to 8 bit #recomment

    #leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1) #perform rectification
    left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR)
    #rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)
    right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR)

    for y in range(0, 600, 30): #Draw lines to validate rectification
        line_thickness = 1
        cv2.line(left_rectified, (0, y), (800, y), (0, 255, 0), thickness=line_thickness)
        cv2.line(right_rectified, (0, y), (800, y), (0, 255, 0), thickness=line_thickness)

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