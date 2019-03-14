import cv2
import numpy as np

vertices = np.array([[0, 385], [0, 275], [450, 250], [640, 275], [640, 385]])
og_image = cv2.imread('1386.bmp', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Grayscale Image", og_image)
cv2.waitKey(0)
normalized_image = np.subtract(np.divide(np.array(og_image).astype(np.float32), 255.0), 0.5)
cv2.imshow("Normalized Image", normalized_image)
cv2.waitKey(0)

blurred_image = cv2.GaussianBlur(og_image, (3, 3), 0)
edge_image = cv2.Canny(blurred_image, 50, 150)
cv2.imshow("Edge Detected", edge_image)
cv2.waitKey(0)


def roi(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


roi_image = roi(edge_image, [vertices])
cv2.imshow("ROI", roi_image)
cv2.waitKey(0)