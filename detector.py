#Deteccion de espermatozoides en video
#Importando librerias
import cv2 as cv
import imutils
import numpy as np

#Cargamos el video a analizar
cap = cv.VideoCapture("videos/video_2.mp4")

#Inicializamos los metodos para eliminar el bg
#fgbg = cv.bgsegm.createBackgroundSubtractorMOG2() #<-- mascara 1
fgbg = cv.createBackgroundSubtractorMOG2() #<--- mascara 2, nota: esta dio mejor redimiento
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3)) #<--- kernel/grid que hara operaciones en los pixeles


#Obteneoms medidas del frame para guardar el video nota: aun no funciona
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
#result = cv.VideoWriter('filename.avi', cv.VideoWriter_fourcc(*'MJPG'),10, size)
result = cv.VideoWriter('output.avi', -1, 20.0, size)

#Loop del video
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width = 1080)
    if ret == False: break

    #Transformamos la imgen a escala de grises para implementar los flitros
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #Aplicacion de los filtros
    fgmask = fgbg.apply(frame)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    fgmask = cv.dilate(fgmask, None, iterations=2)

    #Buscamos los contornos de las figuras resaltadas en los filtros
    cnts = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

    #le agregamos un margen verde a todas las figuras que cumplan el condicional
    for cnt in cnts:
        if cv.contourArea(cnt) > 20:
            x,y,w,h = cv.boundingRect(cnt)
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    #Mostramos la imagen con los margenes asi como el filtro en otra ventana
    #Mostramos el filtro en otra ventana para analizar lo que se esta detectando a fondo
    #cv.drawContours(-1, (0, 255, 0), 2)
    result.write(frame)
    cv.imshow("Frame", frame)
    cv.imshow("Fgmask", fgmask)

    #Esta parte sirve para decirle al programa que cierre le venta precionando ESC
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

#Cierre de las ventanas
cap.release()
result.release()
cv.destroyAllWindows()