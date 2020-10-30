import numpy as np
from PIL import ImageGrab
import time
import cv2
from utils import lane_detection
import pyautogui


while True:
    last_time = time.time()
    screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
    new_screen = lane_detection.find_lane(screen, infer_lines=True)
    # new_screen = lane_detection.color_frame_pipeline(screen, infer_lines=True)
    print(f'Look took {time.time()-last_time} seconds')
    cv2.imshow('window', new_screen)
    # cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(25) and 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
