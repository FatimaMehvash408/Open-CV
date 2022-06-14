import cv2
import matplotlib.pyplot as plt

cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#open cv has pre-trained models for face, eye etc detection
#several positive and negative samples to train the model (Viola-James algorithm)

image=cv2.imread('mehvash.jpg')
#Opencv deals with BGR but matplotlib deals with RGB

#convert to grayscale
gray_image=cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

#
# Min neighbours specify how many neighbors each candidate rectangel should have to retain it
detected_faces=cascade_classifier.detectMultiScale(gray_image,scaleFactor=1.1 , minNeighbors=10)
#print(detected_faces)
for(x, y, width, height) in detected_faces:
    cv2.rectangle(image, (x, y), (x+width, y+height),(0,0,255), 10)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
#plt.imshow(gray_image, cmap='gray')

plt.show()