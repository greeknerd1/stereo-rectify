from plantcv import plantcv as pcv
import cv2
import numpy as np
import os

import pyk4a
from pyk4a import Config, PyK4A

# NFOV_2X2BINNED = 1
#     NFOV_UNBINNED = 2
#     WFOV_2X2BINNED = 3
#     WFOV_UNBINNED = 4
#     PASSIVE_IR = 5

def main():
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()

    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510

    # while 1:
    #     capture = k4a.get_capture()
    #     if np.any(capture.color):
    #         # cv2.imshow("k4a", capture.color[:, :, :3])
    #         # cv2.imshow("k4a", capture.depth)
    #         cv2.imshow("k4a", capture.ir)
    #         key = cv2.waitKey(10) # click any key to quit script
    #         if key != -1:
    #             cv2.destroyAllWindows()
    #             break
    # k4a.stop()

    # cv2.imshow("k4a", capture.color[:, :, :3])
    # cv2.imshow("k4a", capture.depth)
    # cv2.imshow("k4a", capture.ir)

    directory = r'C:\Users\OpenARK\Desktop\stereo-rectify\example\best_image_callibration'
    os.chdir(directory)
    i = 0
    cv2.namedWindow("test")
    while True:
        capture = k4a.get_capture()

        r_img = capture.color[:, :, 2]
        r_img_equalized = pcv.hist_equalization(r_img)

        cv2.imshow('r-img', r_img)
        cv2.imshow('Equalized r-img', r_img_equalized)

        raw_ir_16 = capture.ir
        raw_ir_8 = (raw_ir_16 / 256).astype(np.uint8)

        # cv2.imshow('raw IR 16', raw_ir_16)
        cv2.imshow('raw IR 8', raw_ir_8)

        #We can see up close objects, but farther depth is fully black
        ir_8_equalized = pcv.hist_equalization(raw_ir_8)
        cv2.imshow('IR 8 Equalized', ir_8_equalized)

        #Pretty good and resembles Kinect SDK's viewer the most
        ir_clipped_scaled_16 = np.clip(raw_ir_16, 0, 655 * 1) * 100
        ir_clipped_scaled_8 = (ir_clipped_scaled_16 / 256).astype(np.uint8)
        cv2.imshow('raw IR 8 Clipped Scaled', ir_clipped_scaled_8)

        #Up close objects are saturated, but we can see at a farther depth much better, but has a lot of noise
        ir_clipped_scaled_equalized_8 = pcv.hist_equalization((ir_clipped_scaled_16 / 256).astype(np.uint8)) #hist takes uint8, not uint16
        cv2.imshow('raw IR 8 Clipped Scaled Equalized', ir_clipped_scaled_equalized_8)



        # sift = cv2.xfeatures2d.SIFT_create()
        # kp, des = sift.detectAndCompute(r_img_equalized, None)
        # sift_r_img = cv2.drawKeypoints(r_img_equalized, kp, None)

        # sift2 = cv2.xfeatures2d.SIFT_create()
        # kp2, des2 = sift2.detectAndCompute(ir_clipped_8_equalized, None)
        # sift_ir_8_equalized = cv2.drawKeypoints(ir_clipped_8_equalized, kp2, None)


        # cv2.imshow("R channel Equalized Keypoints", sift_r_img)
        # cv2.imshow("IR 8 Clipped Scaled Equalized Keypoints", sift_ir_8_equalized)








        #Bad
        # ir_8_clipped_equalized = pcv.hist_equalization(np.clip(0, 255, raw_ir_16).astype(np.uint8))
        # cv2.imshow('IR 8 Clipped Equalized', ir_8_clipped_equalized)

        #Bad
        # ir_scaled_16_scaled = np.clip(raw_ir_16 * 100, 0, 65535)
        # cv2.imshow('raw IR 16 Scaled', ir_scaled_16_scaled)

        

        #Bad
        # ir_scaled_8_equalized = pcv.hist_equalization((ir_scaled_16_scaled / 256).astype(np.uint8))
        # cv2.imshow('raw IR 16 Scaled Equalized', ir_scaled_8_equalized)

        


        #Resembles raw IR 8
        # ir_8_normalized = cv2.normalize(raw_ir_8, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # cv2.imshow('Normalized and Equalized IR 8', ir_8_normalized)






        # ir_scaled = ((capture.ir * 100) / 256).astype(np.uint8)
        # ir_normalized = pcv.hist_equalization((capture.ir / 256).astype(np.uint8))
        # ir_clipped_normalized = pcv.hist_equalization(np.clip(0, 255, capture.ir).astype(np.uint8))
        # ir_scaled_normalized = pcv.hist_equalization(((capture.ir * 100) / 256).astype(np.uint8))

        # cv2.imshow("PassiveIR Raw", raw_ir)
        # cv2.imshow("PassiveIR_Scaled", ir_scaled)
        # cv2.imshow("PassiveIR_Normalized", ir_normalized)
        # cv2.imshow("PassiveIR_Clipped_Normalized", ir_clipped_normalized)
        # cv2.imshow("PassiveIR_Scaled_Normalized", ir_scaled_normalized)

        #scaled_ir = np.clip(capture.ir, 0, 655 * 1) * 100
        # hist_scaled_ir = pcv.hist_equalization(((np.clip(capture.ir, 0, int(65535 / 50)) * 50) / 256).astype(np.uint8))
        #hist_scaled_ir = pcv.hist_equalization((scaled_ir / 256).astype(np.uint8)) #hist takes uint8, not uint16


        #cv2.imshow("R-channel color img", r_img)
        #cv2.imshow("Scaled IR", scaled_ir)
        #cv2.imshow("Scaled then Normalized IR", hist_scaled_ir)

        # sift = cv2.xfeatures2d.SIFT_create()
        # kp, des = sift.detectAndCompute(hist_scaled_ir, None)
        # sift_hist_scaled_ir = cv2.drawKeypoints(hist_scaled_ir, kp, hist_scaled_ir)

        # sift2 = cv2.xfeatures2d.SIFT_create()
        # kp2, des2 = sift2.detectAndCompute(r_img, None)
        # sift_r_img = cv2.drawKeypoints(r_img, kp2, None)


        # cv2.imshow("R channel Img With Keypoints", sift_r_img)
        # cv2.imshow("Scaled Normalized IR with Keypoints", sift_hist_scaled_ir)






        key = cv2.waitKey(1)
        if key % 256 == 27: # ESC pressed
            print('Exiting!')
            break
        elif key % 256 == 32: # SPACE pressed save photo

            cv2.imwrite('color-' + str(i) + '.png', r_img_equalized)
            cv2.imwrite('ir-' + str(i) + '.png', ir_8_equalized)

            # cv2.imwrite("PassiveIR_Raw.png", raw_ir)
            # cv2.imwrite("PassiveIR_Scaled.png", ir_scaled)
            # cv2.imwrite("PassiveIR_Normalized.png", ir_normalized)
            # cv2.imwrite("PassiveIR_Clipped_Normalized.png", ir_clipped_normalized)
            # cv2.imwrite("PassiveIR_Scaled_Normalized.png", ir_scaled_normalized)
            

            # cv2.imwrite('color-' + str(i) + '.png', capture.color[:, :, :3])
            # cv2.imwrite('ir-' + str(i) + '.png', capture.ir)
            i += 1
            print('Image saved!')
     
    k4a.stop()


if __name__ == "__main__":
    main()
