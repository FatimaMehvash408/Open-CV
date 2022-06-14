import cv2
import numpy as np

def draw_the_lines(image, lines):
    # all 0 means black image
    lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    # there are x and y coordinates for strt and end

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=3)

        # finally we have to merge image with line
        image_with_lines = cv2.addWeighted(image, 0.8, lines_image, 1, 0.8)

        return image_with_lines


def region_of_interest(image, region_points):
    # we are going to replace pixels with 0(black)- the region we are not interested def get_detected_lanes(image):
    mask = np.zeros_like(image)

    # region that we are interested in has value 255
    cv2.fillPoly(mask, region_points, 255)

    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


def get_detected_lanes(image):
    (height, width) = (image.shape[0], image.shape[1])

    # we have to turn image into grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection kernel
    canny_image = cv2.Canny(gray_image, 180, 128)

    # we are interested in the lower region of the image (i.e is the driving lane)
    region_of_interest_vertices = [
        (width / 8, height),
        (width / 2, height * 0.5),
        (width * 3 / 4, height)
    ]
    # we can get rid of un relevant part of imaging using masking
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))
    # using line detection algorithm
    # angle in radians
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180, threshold=50, lines=np.array([]),
                            minLineLength=40, maxLineGap=150)

    image_with_lines = draw_the_lines(image, lines)
    return image_with_lines


# video=several frames
video = cv2.VideoCapture('lane_video.mp4')
while video.isOpened():

    is_grabbed, frame = video.read()

    if not is_grabbed:
        break
    frame = get_detected_lanes(frame)
    # cv2.imshow('lane_video', video)
    cv2.imshow('lane_video', frame)
    cv2.waitKey(5)
video.release()
cv2.destroyWindow()
