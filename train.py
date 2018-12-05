import cv2
import os
import face_recognition

imagePaths = os.listdir('known_faces/')

known_encodings = []
known_names = []

for l in imagePaths:
    for k in os.listdir('known_faces/'+l):
        print("Encoding image: "+l+"/"+k)
        
        image = cv2.imread('known_faces/'+l+'/'+k)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = face_recognition.face_locations(rgb, model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        for e in encodings:
            known_encodings.append(e)
            known_names.append(l)

f,g = open('known_encodings.txt', 'w'), open('known_names.txt', 'w')
c,n = "",""
for a in range(len(known_encodings)):
    for b in known_encodings[a]:
        c = c + str(b) + '\n'
    c = c + '\n\n'
    n = n + known_names[a] + '\n'
f.write(c.strip())
g.write(n.strip())
f.close()
g.close()