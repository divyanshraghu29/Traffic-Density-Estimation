import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib
base_path = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(
	base_path, "empty.jpg"
)
video_path = os.path.join(base_path, "trafficvideo.mp4")


input_img=cv2.imread("img_path")

cap= cv2.VideoCapture('video_path')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

img_coord=np.float32([[976,223],[1264,220],[453,863],[1473,865]])

new_coor=np.float32([[472,52],[800,52],[472,830],[800,830]])
#dst_points({Point2f(472,52),Point2f(472,830),Point2f(800,830),Point2f(800,52)})
p,s=cv2.findHomography(img_coord,new_coor)
#homographic transformtaion of background image
perspective_bg=cv2.warpPerspective(input_img,p,(1080,1920))
perspective_bg_gray=cv2.cvtColor(perspective_bg, cv2.COLOR_BGR2GRAY)
cropped_image_bg = perspective_bg_gray[52:831, 472:801]

time1=0
x_=[]
y_=[]
plt.ion()
fig, ax = plt.subplots(figsize=(5,5))
line, = ax.plot(0, 0)
plt.title("density_plot", fontsize=20)
plt.xlabel("X-axis-time_frame")
plt.ylabel("Y-axis-pixel_density")
while cap.isOpened():

        ret, frame= cap.read()        

        gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        perspective=cv2.warpPerspective(gray,p,(1080,1920))
        cropped_image=perspective[52:831, 472:801]
        matA=cv2.absdiff(cropped_image, cropped_image_bg)
        pixel_density= np.sum(matA>25)
        time1+=1
        x_.append(time1)
        y_.append(pixel_density/(matA.shape[0]*matA.shape[1]))
        line.set_xdata(x_)
        line.set_ydata(y_)
        fig.canvas.draw()
        fig.canvas.flush_events()
        cv2.imshow('Cars',cropped_image)
        plt.plot(x_,y_)
        plt.show()

        if cv2.waitKey(1) == 13:
            break
plt.draw()
cap.release()
cv2.destroyAllWindows()
