import cv2
from random import randrange

#Machine learning opencv (Pre-Built)Neural Network for detecting faces
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to detect faces in
img = cv2.imread('mens.jpg')

#Must convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#Draw Rectangles around the faces
for (x, y, w, h)  in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)


#displaying image
cv2.imshow('FACE DETECTOR' , img)

#processing time
cv2.waitKey()
print("Code Completed")
