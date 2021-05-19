import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

height = 1210
width = 1915

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
    
region_of_interest_vertices = [
    (0, height),
    (width / 2, height/100),
    (width, height),
    ]

image = mpimg.imread('/home/edvard/Pictures/lanes2.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#gray_image = cv2.GaussianBlur(gray_image, (3,3), 0)
cannyed_image = cv2.Canny(gray_image, 100, 200)
plt.imshow(cannyed_image)
cv2.waitKey(20)
cropped_image = region_of_interest(
    cannyed_image,
    np.array([region_of_interest_vertices], np.int32),
)
plt.figure()
plt.imshow(cropped_image)


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return    # Make a copy of the original image.
    img = np.copy(img)    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)    # Return the modified image.
    return img

lines = cv2.HoughLinesP(
    cropped_image,
    rho=6,
    theta=np.pi / 60,
    threshold=160,
    lines=np.array([]),
    minLineLength=40,
    maxLineGap=25
)

left_line_x = []
left_line_y = []
right_line_x = []
right_line_y = []

for line in lines:
    for x1, y1, x2, y2 in line:
        slope = (y2 - y1) / (x2 - x1)  # <-- Calculating the slope.
        if math.fabs(slope) < 0.8:  # <-- Only consider extreme slope
            continue
        if slope <= 0:  # <-- If the slope is negative, left group.
            left_line_x.extend([x1, x2])
            left_line_y.extend([y1, y2])
        else:  # <-- Otherwise, right group.
            right_line_x.extend([x1, x2])
            # <-- Just below the horizon
            right_line_y.extend([y1, y2])

min_y = image.shape[0] * (3 / 5)
max_y = image.shape[0]

poly_left = np.poly1d(np.polyfit(
    left_line_y,
    left_line_x,
    deg = 1
))
left_x_start=int(poly_left(max_y))
left_x_end=int(poly_left(min_y))

poly_right=np.poly1d(np.polyfit(
    right_line_y,
    right_line_x,
    deg = 1
))

right_x_start=int(poly_right(max_y))
right_x_end=int(poly_right(min_y))

line_image=draw_lines(
    image,
    [[
        [left_x_start, max_y, left_x_end, min_y],
        [right_x_start, max_y, right_x_end, min_y],
    ]],
    thickness=5,
)
plt.figure()
plt.imshow(line_image)
plt.show()
