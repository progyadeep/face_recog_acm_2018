import cv2
import face_recognition
import numpy
import os

known_encodings, known_names = [],[]

f = open('known_encodings.txt', 'r')
k_e = f.read().split("\n\n")
for k in k_e:
    t = k.strip().split()
    t = [float(tt) for tt in t]
    known_encodings.append(numpy.array(t))

f.close()
f = open('known_names.txt', 'r')
known_names = f.read().strip().split("\n")
f.close()


testImage = cv2.cvtColor(cv2.imread(input("Path of test image: ")), cv2.COLOR_BGR2RGB)
print("Encoding test image")
test_box = face_recognition.face_locations(testImage, model="cnn")
test_enc = face_recognition.face_encodings(testImage, test_box)

print("Recognizing the ugly face")

matches = []
for k in range(len(known_encodings)):
    matches.append([face_recognition.compare_faces(known_encodings[k], test_enc)[0], known_names[k]])


for m in matches:
    if m[0] == True:
        print(m[1])
        exit()

print("Face not recognized.")