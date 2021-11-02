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

    directory = r'C:\Users\OpenARK\Desktop\pyk4a\example\images'
    os.chdir(directory)
    i = 0
    cv2.namedWindow("test")
    while True:
        capture = k4a.get_capture()
        cv2.imshow("test", capture.ir)
        key = cv2.waitKey(1)
        if key % 256 == 27: # ESC pressed
            print('Exiting!')
            break
        elif key % 256 == 32: # SPACE pressed save photo
            cv2.imwrite('color-' + str(i) + '.png', capture.color[:, :, :3])
            cv2.imwrite('ir-' + str(i) + '.png', capture.ir)
            i += 1
            print('Image saved!')
     
    k4a.stop()


if __name__ == "__main__":
    main()
