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
    config = Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.PASSIVE_IR,
            synchronized_images_only=True,
        )
    k4a = PyK4A(config)
    #k4a.start()


    k4a.load_calibration_json('calibration_data')
    print(k4a.calibration_data)
     
    #k4a.stop()


if __name__ == "__main__":
    main()
