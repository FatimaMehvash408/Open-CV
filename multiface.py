import cv2
import matplotlib.pyplot as plt

cascade_classifier=cv2.CascadeClassifier_convert('haarcascade_frontalface_alt.xml')
image=cv2.imread('people.jpg')
gray_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detected_faces=cascade_classifier.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=10)
for (x, y, width, height) in detected_faces:
    cv2.rectangle(image, x,y,(x+width, y+height), (0,0,255), 10)

plt.show(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
plt.show()

