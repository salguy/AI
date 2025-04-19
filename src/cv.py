import cv2
import numpy as np

def load_image(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def detect_all_boxes(gray, min_area=30, max_area=10000):
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if len(approx) == 4 and min_area < area < max_area:
            boxes.append(approx)
    return boxes

def classify_boxes(boxes, image_shape):
    h, w = image_shape[:2]
    outer, inner = [], []

    for box in boxes:
        M = cv2.moments(box)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if (cx < w * 0.2 or cx > w * 0.8) and (cy < h * 0.2 or cy > h * 0.8):
            outer.append(box)
        else:
            inner.append(box)
    return outer, inner

def detect_main_inner_rectangle(gray, min_area=10000):
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest = None
    max_area = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if len(approx) == 4 and area > max_area and area > min_area:
            largest = approx
            max_area = area
    return largest
