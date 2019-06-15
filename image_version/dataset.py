''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
	==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
	==> Each face will have a unique numeric integer ID as 1, 2, 3, etc                       
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    
Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18    
'''

import cv2
import os
from PIL import Image
import glob

#temp = []
db = open("database.txt", "a+")

face_detector = cv2.CascadeClassifier('C:/Users/soyoo/Desktop/Final/image_version/haarcascade_frontface.xml')
name = input("\nwhat is your name? ==> ")
db.write(name + "\n")
db.close()

db = open("database.txt", "r")
f1 = db.readlines()
count = 0
for x in f1:
    count += 1;
face_id = count
db.close()

# Initialize individual sampling face count
count = 0

image_list = []
root = input('Enter the folder location: ')
for filename in glob.glob(root+'/*.jpg'):
    im = cv2.imread(filename)
    im = im.astype('uint8')
    image_list.append(im)
    #img = cv2.flip(img, -1) # flip video image vertically
for img in image_list:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        cv2.imwrite("dataset/" + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
    print('Processing... image'+ str(count))
    if count >= 30:
        print("\n [INFO] Exiting Program and cleanup stuff")
        cv2.destroyAllWindows()
        break