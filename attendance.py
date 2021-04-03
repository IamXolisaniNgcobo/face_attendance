import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'
images = []
class_names = []
my_list = os.listdir(path)
print(my_list)

for image_name in my_list:
    current_image = cv2.imread(f'{path}/{image_name}')
    images.append(current_image)
    class_names.append(os.path.splitext(image_name)[0])
print(class_names)


def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode_img = face_recognition.face_encodings(img)
        encode_list.append(encode_img)

    return encode_list


def mark_attendance(name):
    with open("Attendance_data.csv", "r+") as f:
        data_list = f.readlines()
        name_list = []
        for line in data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            date_string = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{date_string}")


encode_known_list = find_encodings(images)
print("Encoding Complete")

video_capture = cv2.VideoCapture(0)
while True:
    success, img = video_capture.read()
    small_img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

    faces_current_frame = face_recognition.face_locations(small_img)
    encode_current_frame = face_recognition.face_encodings(small_img, faces_current_frame)

    for encode_face, face_loc in zip(encode_current_frame, faces_current_frame):

        matches = face_recognition.compare_faces(encode_known_list[0], encode_face)
        face_dist = face_recognition.face_distance(encode_known_list[0], encode_face)

        print(face_dist)
        match_index = np.argmin(face_dist)

        if matches[match_index]:
            name = class_names[match_index].upper()
            print(name)
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            mark_attendance(name)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)
