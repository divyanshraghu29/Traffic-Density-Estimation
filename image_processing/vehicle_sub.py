
from numpy.core.fromnumeric import resize
import cv2
import numpy as np
from time import sleep

import matplotlib.pyplot as plt
import time
import matplotlib
from matplotlib.animation import FuncAnimation

largura_min=80 #Largura minima do retangulo
altura_min=80 #Altura minima do retangulo
offset=6 #Erro permitido entre pixel  
pos_linha=750 
upper=200 #Posição da linha de contagem 

delay= 60 #FPS do vídeo

detec = []
carros= 0	
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
h,w,k=input_img.shape

cap= cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))

frame_height = int(cap.get(4))



subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()

img_coords=np.array([[980,200],[1260,200],[1260,400],[800,400]])

img_coord=np.array([[980,200],[1260,200],[800,400],[1260,400]])

new_coor=np.float32([[472,100],[1000,100],[472,400],[1000,400]])

p,s=cv2.findHomography(img_coord,new_coor)


while cap.isOpened():
    
    ret , frame1 = cap.read()      

    gray= cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    perspective=cv2.warpPerspective(gray,p,(1920,1080))

    src=perspective[100:401, 450:1001]

    

    blur = cv2.GaussianBlur(src,(3,3),5)
    img_sub = subtracao.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilatada = cv2.morphologyEx (dilatada, cv2. MORPH_CLOSE , kernel)
    contorno,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(src, (80, 40), (600, 40), (255,127,0), 3) 
    for(i,c) in enumerate(contorno):
        (x,y,w,h) = cv2.boundingRect(c)
        validar_contorno = (w >= largura_min) and (h >= altura_min)
        if not validar_contorno:
            continue

        cv2.rectangle(src,(x,y),(x+w,y+h),(0,255,0),2)        
        centro = pega_centro(x, y, w, h)
        detec.append(centro)
       # cv2.circle(frame1, centro, 4, (0, 0,255), -1)

        for (x,y) in detec:

            if y<(40+offset) and y>(40-offset) :  ##checking if the vehicles have reached the bar line i.e they have reached the traffic cross and subtracting them from the queue

                carros-=1
                cv2.line(frame1, (553, pos_linha), (1500, pos_linha), (0,127,255), 3) 
                cv2.line(src, (26, upper), (1500, upper), (0,127,255), 3)  
                detec.remove((x,y))
        
       
    cv2.putText(frame1, "VEHICLE COUNT : "+str(carros), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , src)
    cv2.imshow("Video Original1" , frame1)
    cv2.imshow("Detectar",dilatada)
    
    ##writing the vehicle count in a file
    with open("count.txt", "a") as file1:
        file1.write(str(carros)+'\n')
    
    if cv2.waitKey(1) == 27:
        break
file1.close()   

cap.release()
cv2.destroyAllWindows()
