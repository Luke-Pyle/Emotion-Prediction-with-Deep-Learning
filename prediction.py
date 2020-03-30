import numpy as np
import cv2
import cv2
import tensorflow as tf
import keras
import numpy as np
import pygame 

ID = 0
CATEGORIES = ["Happy", "Sad"]

pygame.init()

X = 600
Y = 600

white = (255, 255, 255)

happy_image = pygame.image.load("happy_emoji.png")
sad_image = pygame.image.load("sad_emoji.png")

display_surface = pygame.display.set_mode((X, Y )) 

def prepare(filepath):
    IMG_SIZE = 100
    #img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(filepath, (IMG_SIZE, IMG_SIZE))
    new_array =  cv2.cvtColor(new_array,cv2.COLOR_BGR2GRAY)
    cv2.imshow('', new_array)
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = keras.models.load_model("MODEL")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1)

while 1:
    display_surface.fill(white) 
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h]
        roi_color = img[y:y+h, x:x+w]
        prediction = model.predict([prepare(roi_color)])
        
        if np.round(prediction[0][0]) == 1:
              display_surface.blit(sad_image, (0, 0)) 
        else:
              display_surface.blit(happy_image, (0, 0)) 
        
        #print(np.round(prediction))
        #cv2.imwrite(str(ID) +'Test.jpg', roi_color)
        #ID += 1

        #if x < 200 or x > 280 and y < 160 or y > 260:
            #print("Move!")

    cv2.imshow('img',img)

    for event in pygame.event.get() : 

        if event.type == pygame.QUIT : 
            pygame.quit() 

            quit() 
    pygame.display.update()
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()






 
     




