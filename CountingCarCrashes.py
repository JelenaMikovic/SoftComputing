import numpy
import sys
import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

nbins = 9
cell_size = (8, 8) 
block_size = (3, 3) 

hog = cv2.HOGDescriptor(_winSize=(120 // cell_size[1] * cell_size[1], 
                                  60 // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)

positive_features = []
negative_features = []
labels = []

data_path = sys.argv[1]

image_path = data_path + "pictures"

for i in range(1, 223):
    img = cv2.imread(image_path + "/p_" + str(i) + ".png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog.compute(gray)
    positive_features.append(features)
    labels.append(1)

for i in range(1, 301):
    img = cv2.imread(image_path + "/n_" + str(i) + ".png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog.compute(gray)
    negative_features.append(features)
    labels.append(0)

x = np.vstack((positive_features, negative_features))
y = np.array(labels)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# print('Train shape: ', x_train.shape, y_train.shape)
# print('Test shape: ', x_test.shape, y_test.shape)

clf_svm = SVC(kernel='linear', probability=True) 
clf_svm.fit(x_train, y_train)
y_train_pred = clf_svm.predict(x_train)
y_test_pred = clf_svm.predict(x_test)
# print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
# print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))

red_line_coordinates = (1080, 0, 1080, 450)


def count(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    #cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

    car_count = 0
    prev_car_count = 0
    cooldown = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break 

        min_width = 10
        min_height = 10
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh, frame_bin = cv2.threshold(frame_gray, 150, 255, cv2.THRESH_BINARY)
        frame_numbers = cv2.dilate(frame_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
        contours, _ = cv2.findContours(frame_numbers.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = [cv2.boundingRect(contour) for contour in contours]


        if(cooldown == 0):
            for contour in rectangles:
                x, y, w, h = contour
                if w > min_width and h > min_height:

                    roi = frame[y:y+h, x:x+w]
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    resized_roi = cv2.resize(gray_roi, (120, 60), interpolation=cv2.INTER_NEAREST) 

                    hog_features = hog.compute(resized_roi)
                    hog_features = hog_features.reshape(1, -1)

                    prediction = clf_svm.predict(hog_features)
                    
                    if prediction == 1 and check_intersection((x, y, x + w, y + h), red_line_coordinates) :
                        car_count += 1
                        cooldown = 20
        else:
            cooldown -= 1        

        #cv2.putText(frame, f"Total Cars: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #cv2.imshow("Video", frame)

        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return car_count

def check_intersection(bbox, line):
    x1, y1, x2, y2 = line
    x, y, xw, yh = bbox
    return (x <= x2 and xw >= x1 and y <= y2 and yh >= y1)

actual_values = []

video_path = data_path + "videos/v1.mp4"
c = count(video_path)
print("v1.mp4-5-", c)
actual_values.append(c)

video_path = data_path + "videos/v2.mp4"
c = count(video_path)
print("v2.mp4-6-", c)
actual_values.append(c)

video_path = data_path + "videos/v3.mp4"
c = count(video_path)
print("v3.mp4-1-", c)
actual_values.append(c)

predicted_values = np.array([5, 6, 1])
mae = np.mean(np.abs(np.array(actual_values) - predicted_values))

print(f"Mean Absolute Error (MAE): {mae}")