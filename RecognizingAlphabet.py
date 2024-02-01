import csv
import cv2
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import collections
import math
from scipy import ndimage
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
from sklearn.cluster import KMeans

data_path = sys.argv[1]
images_folder = os.path.join(data_path, 'pictures')
csv_file = os.path.join(data_path, 'res.csv')

unique_chars = set()
data = []

with open(csv_file, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader) 
    for row in reader:
        image_path, true_word = row
        data.append({'image': image_path, 'true_word': true_word})
        word = true_word.replace(" ", "")
        for c in word:
            unique_chars.add(c)

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def display_image(image, color=False):
    plt.figure()
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')
    plt.show()

def invert(image):
    return 255-image

def dilate(image):
    kernel = np.ones((2, 2)) 
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((3, 3)) 
    return cv2.erode(image, kernel, iterations=1)

def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)

def select_roi(image_orig, image_bin):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 20 and area < 2000 and h < 100 and h > 10 and w > 10:
            region = image_bin[y:y+h+1, x:x+w+1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    regions_array = sorted(regions_array, key=lambda region: region[1][0])

    merged_regions = []
    current_region = regions_array[0]

    for next_region in regions_array[1:]:
        x1, y1, w1, h1 = current_region[1]
        x2, y2, w2, h2 = next_region[1]

        if ((x1 >= x2 and x1 <= x2+w2 and x1+w1 >= x2 and x1+w1 >= x2+w2) or (x2 >= x1 and x2 <= x1+w1 and x2+w2 >= x1 and x2+w2 <= x1+w1)):
            merged_x = min(x1, x2)
            merged_y = min(y1, y2)
            merged_w = max(x1 + w1, x2 + w2) - merged_x
            merged_h = max(y1 + h1, y2 + h2) - merged_y
            merged_image = np.vstack((current_region[0], next_region[0]))
            current_region[0] = cv2.resize(merged_image, (28, 28))
            current_region[1] = (merged_x, merged_y, merged_w, merged_h)
        else:
            x1, y1, w1, h1 = current_region[1]
            merged_regions.append(current_region)
            current_region = next_region
    
    merged_regions.append(current_region)

    #for region in merged_regions:
    #    x, y, w, h = region[1]
    #    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    sorted_regions = [region[0] for region in merged_regions]
    sorted_rectangles = [region[1] for region in merged_regions]
    region_distances = []

    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances

alphabet = sorted(list(unique_chars))
all_regions = []
letters = []
all_distances = [] 

def prepare_training_data():
    for row in data:
        full_path = os.path.join(images_folder, row["image"])
        image = load_image(full_path)
        x1, y1, x2, y2 = 250, 170, 830, 250
        cropped_image = image[y1:y2, x1:x2]
        img_gray = image_gray(cropped_image)
        img_bin = image_bin(img_gray)
        img_bin = erode(dilate(img_bin))
        selected_regions, numbers, distances  = select_roi(cropped_image.copy(), img_bin)
        #display_image(selected_regions)
        all_regions.append(numbers)
        word = row["true_word"].replace(" ", "")
        for c in word:
            letters.append(alphabet.index(c))

def scale_to_range(image):
    return image/255

def matrix_to_vector(image):
    return image.flatten()

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        for num in region:
            scale = scale_to_range(num)
            ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann

def convert_output(letters):
    nn_outputs = []
    for letter in letters:
        output = np.zeros(len(alphabet))
        output[letter] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann

def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)
    
    print("\nTraining started...")
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)
    print("\nTraining completed...")
    return ann

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

prepare_training_data()
inputs = prepare_for_ann(all_regions)
outputs = convert_output(letters)
ann = create_ann(output_size=len(alphabet))
ann = train_ann(ann, inputs, outputs, epochs=1000)

def display_result_with_spaces(outputs, alphabet, k_means):
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])
    space = False
    if w_space_group[1][0] > 20:
        space = True

    result = alphabet[winner(outputs[0])]
    
    for idx, output in enumerate(outputs[1:, :]):
        if k_means.labels_[idx] == w_space_group[0] and space:
            result += ' '
        result += alphabet[winner(output)]
    
    return result

def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        min_length = min(len(str1), len(str2))
        distance = sum(c1 != c2 for c1, c2 in zip(str1[:min_length], str2[:min_length]))
        distance += abs(len(str1) - len(str2))
    else:
        distance = sum(c1 != c2 for c1, c2 in zip(str1, str2))
    
    return distance

hamming = 0
k_means = KMeans(n_clusters=2, n_init=10)

for row in data:
    full_path = os.path.join(images_folder, row["image"])
    image = load_image(full_path)
    x1, y1, x2, y2 = 250, 170, 830, 250
    cropped_image = image[y1:y2, x1:x2]
    img_gray = image_gray(cropped_image)
    img_bin = image_bin(img_gray)
    img_bin = erode(dilate(img_bin))
    selected_regions, numbers, distances = select_roi(cropped_image.copy(), img_bin)
    test_outputs = []
    test_outputs.append(numbers)
    test_inputs = prepare_for_ann(test_outputs)
    result = ann.predict(np.array(test_inputs, np.float32))
    distances = np.array(distances).reshape(len(distances), 1)
    k_means.fit(distances)
    word = display_result_with_spaces(result, alphabet, k_means)
    print(row["image"] + "-" + row["true_word"] + "-" + word)
    hamming += hamming_distance(row["true_word"], word)

print("Sum of Hamming distances: " + str(hamming))