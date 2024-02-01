import numpy as np
import cv2
from matplotlib import pyplot as plt
import csv
import sys

def load_image(path):
    return cv2.imread(path)

def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
        
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return image_bin
    
def count_dittos(image_path):
    img = load_image(image_path)

    img_bin = cv2.adaptiveThreshold(image_gray(img), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations = 3)
    sure_bg = cv2.dilate(opening, kernel, iterations = 3)
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dark_purple = np.array([120, 0, 0])
    light_purple = np.array([140, 245, 255])
    
    mask = cv2.inRange(img_hsv, dark_purple, light_purple)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 2)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 2)
    
    combined_mask = cv2.bitwise_and(mask, cv2.bitwise_not(sure_bg))
    #display_image(combined_mask)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_ditto_area = 800 
    min_ditto_aspect_ratio = 0.75

    dittos = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_ditto_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if aspect_ratio > min_ditto_aspect_ratio:
                dittos += 1

    return dittos

actual_counts = []
expected_counts = []
path = sys.argv[1]
image_path = sys.argv[1]
csv_file = sys.argv[1] + "ditto_count.csv"

with open(csv_file, 'r', newline='') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        filename = row['Naziv slike']
        dittos_to_count = int(row['Broj ditto-a'])
        image_file_path = image_path + filename
        dittos = count_dittos(image_file_path)
        actual_counts.append(dittos)
        expected_counts.append(dittos_to_count)
        print(filename, "-", dittos_to_count, "-", dittos)

absolute_errors = np.abs(np.array(actual_counts) - np.array(expected_counts))
mae = np.mean(absolute_errors)
print(mae)