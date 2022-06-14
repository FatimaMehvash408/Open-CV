import cv2
import numpy as np

image = cv2.imread('picture2.jpg', cv2.IMREAD_COLOR)

# values close to 8: darker pixels
# values closet to 255: brighter pixel
# print(image.shape)
# print(np.amax(image))
# print(image)
cv2.imshow('Computer Vision', image)
# cv2.waitKey(0)
# cv2.destroyWindow()

# We store R,G,B components on 8 bits

# kernel=np.ones((5,5))
kernel = np.ones((5, 5)) / 25

# BLUR IMAGE KERNEL
blur_image=cv2.filter2D(image, -1, kernel)
# cv2.imshow('original Image', image)
cv2.imshow('Blur Image', blur_image)
# cv2.waitKey(0)
# cv2.destroyWindow()
# print(kernel)


# EDGE KERNEL(Laplacian Kernel)
# we have to transform the image into grayscale

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
#result_image = cv2.filter2D(image, -1, kernel)
result_image=cv2.Laplacian(gray_image, -1)
cv2.imshow('Original_gray Image', gray_image)
cv2.imshow('Result Image', result_image)
#cv2.waitKey(0)
#cv2.destroyWindow()


# SHARPEN OPERATION
#  in face recognition if we have a  blurry image we apply inc precision of the underlying model

kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

sharpen_image = cv2.filter2D(image, -1, kernel)

cv2.imshow('Sharpen Image', sharpen_image)
cv2.waitKey(0)
cv2.destroyWindow()