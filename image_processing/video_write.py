import cv2
import numpy as np
from pathlib import Path
import os


def transformVideo(videoPath):
    cap = cv2.VideoCapture(videoPath)

    ## getting frame dimensions
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    ##declaring the variable to store the transformed frames
    out = cv2.VideoWriter(os.path.dirname(videoPath) + '/transformedVideo.avi',
                          cv2.VideoWriter_fourcc(*'XVID'), 15, (1920, 1080))

    # coordinates of road in real frame

    img_coord = np.array([[980, 200], [1260, 200], [800, 400], [1260, 400]])

    # coordinates of road in transformed frame

    new_coor = np.float32([[472, 100], [1000, 100], [472, 400], [1000, 400]])
    # applying homogra[hic transformation

    p, s = cv2.findHomography(img_coord, new_coor)

    while cap.isOpened():

        ret, frame1 = cap.read()
        if ret == True:
            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            perspective = cv2.warpPerspective(gray, p, (1920, 1080))
            # frame cropping
            src = perspective[100:401, 450:1001]
            # writing frame in the video
            out.write(perspective)
            if cv2.waitKey(1) == 27:
                break
        else:
            break
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    return True
