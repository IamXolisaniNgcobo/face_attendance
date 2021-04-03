import cv2
import numpy as np
import face_recognition

elon_photo = face_recognition.load_image_file('ImagesBasic/Elon Musk.jpg')
elon_photo = cv2.cvtColor(elon_photo, cv2.COLOR_BGR2RGB)
elon_photo_test = face_recognition.load_image_file('ImagesBasic/Bill Gates.jpg')
elon_photo_test = cv2.cvtColor(elon_photo_test, cv2.COLOR_BGR2RGB)

face_location = face_recognition.face_locations(elon_photo)[0]
encode_elon_photo = face_recognition.face_encodings(elon_photo)[0]
cv2.rectangle(elon_photo, (face_location[3], face_location[0]), (face_location[1], face_location[2]), (0, 255, 0), 2)

face_location_test = face_recognition.face_locations(elon_photo_test)[0]
encode_elon_photo_test = face_recognition.face_encodings(elon_photo_test)[0]
cv2.rectangle(elon_photo_test, (face_location_test[3], face_location_test[0]),
              (face_location_test[1], face_location_test[2]), (0, 255, 0), 2)

results = face_recognition.compare_faces([encode_elon_photo], encode_elon_photo_test)
face_distance = face_recognition.face_distance([encode_elon_photo], encode_elon_photo_test)
print(results, face_distance)
print(face_location)
cv2.putText(elon_photo, f'{results}{round(face_distance[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
cv2.imshow('Elon Musk', elon_photo)
cv2.imshow('Elon Test', elon_photo_test)
cv2.waitKey(0)
