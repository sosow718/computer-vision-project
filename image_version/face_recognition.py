''''
Real Time Face Recogition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc
	==> LBPH computed model (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition
Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18
'''

import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('trainer/trainer.yml')
cascadePath = 'C:/Users/soyoo/Desktop/Final/image_version/haarcascade_frontface.xml'
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0
names = []
# names related to ids: example ==> Marcelo: id=1,  etc
db = open("database.txt", "r")
f1 = db.readlines()

for x in f1:
    names.append(x[0:-1])
db.close()

# Initialize and start realtime video capture

imgName = input("Enter the image file name: ")
while True:
    img = cv2.imread(imgName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(3), int(3)),
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        id -= 1

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.namedWindow('Guess who I am', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Guess who I am', 450, 600)
    cv2.imshow('Guess who I am', img, )
    cv2.imwrite(str(id) + '.jpg', img)

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cv2.destroyAllWindows()