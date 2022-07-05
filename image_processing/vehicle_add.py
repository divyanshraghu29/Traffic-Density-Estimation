import cv2
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import time
import matplotlib
from matplotlib.animation import FuncAnimation

lmin=80 #Largura minima do retangulo
amin=80 #Altura minima do retangulo
offset=6 #Erro permitido entre pixel  
pos=750 
upper=200 #Posição da linha de contagem 
delay= 60 #FPS do vídeo

detec = []
detec_trans=[]
count= 0
##finding the center of the rectangle
def pega_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy
base_path = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(
	base_path, "empty.jpg"
)
video_path = os.path.join(base_path, "trafficvideo.mp4")

input_img=cv2.imread(img_path)
cap= cv2.VideoCapture(video_path)

subtract = cv2.bgsegm.createBackgroundSubtractorMOG()

file1 = open("count_+.txt","a")
def pointsAreOnSameSideOfLine(a, b, c, x1, y1, x2, y2):
	fx1 = 0 # Variable to store a * x1 + b * y1 - c
	fx2 = 0 # Variable to store a * x2 + b * y2 - c

	fx1 = a * x1 + b * y1 - c
	fx2 = a * x2 + b * y2 - c

	# If fx1 and fx2 have same sign
	if ((fx1 * fx2) > 0):
		return True

	return False

a1=640
b1=-423
c1=718969

a2=-43
b2=14
c2=-51229

img_coords=np.array([[976,223],[1263,220],[1473,865],[553,863]])
vectorAB=np.array([423,-640])
vectorBA=np.array([-423,640])
vectorCD=np.array([-210,-645])
vectorDC=np.array([210,645])

img_coord=np.array([[980,200],[1260,200],[800,400],[1260,400]])
new_coor=np.float32([[472,100],[1000,100],[472,400],[1000,400]])
p,s=cv2.findHomography(img_coord,new_coor)

while True:
    
    ret , frame1 = cap.read()
    for s in range(len(img_coords)):
        start_X = img_coords[s % len(img_coords)][0]
        start_Y = img_coords[s % len(img_coords)][1]
        end_X = img_coords[(s+1) % len(img_coords)][0]
        end_Y = img_coords[(s+1) % len(img_coords)][1]
       # print(start_X,start_Y,end_X,end_Y)
        cv2.line(frame1, (start_X, start_Y),(end_X, end_Y), [255, 0, 0], 3)
   
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)


    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtract.apply(blur)
    dilate = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx (dilate, cv2. MORPH_CLOSE , kernel)
    dilatada = cv2.morphologyEx (dilatada, cv2. MORPH_CLOSE , kernel)
    contours,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (553, pos), (1500, pos), (255,127,0), 3) 
    cv2.line(frame1, (205, upper), (1500, upper), (255,127,0), 3) 
    for(i,c) in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(c)
        validar_contours = (w >=lmin) and (h >= amin)
        if not validar_contorno:
            continue      
        centro = pega_centro(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame1, centro, 4, (0, 0,255), -1)

        for (x,y) in detec:
     
            if y<(pos+offset) and y>(pos-offset) and pointsAreOnSameSideOfLine(a1,b1,c1,x,y,1125,425) and pointsAreOnSameSideOfLine(a2,b2,c2,x,y,1125,425) and np.dot(vectorAB,[[976-x],[226-y]])>0 and np.dot(vectorBA,[[553-x],[863-y]])>0:
                count+=1
                cv2.line(frame1, (205, pos), (1500, pos), (0,127,255), 3)  
                detec.remove((x,y))
            
                cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),-1) 
  

    with open("count_+.txt", "a") as file1:
      file1.write(str(count)+'\n')     
    cv2.putText(frame1, "VEHICLE COUNT : "+str(count), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame1)
    cv2.imshow("Detectar",dilatada)
        
    if cv2.waitKey(1) == 27:
        break
file1.close()    
cv2.destroyAllWindows()
cap.release()
