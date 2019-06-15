''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
	==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
	==> Each face will have a unique numeric integer ID as 1, 2, 3, etc                       
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    
Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18    
'''

import cv2
import os


#temp = []
db = open("database.txt", "a+")

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_detector = cv2.CascadeClassifier('C:/Users/soyoo/Desktop/Final/realtime_version/haarcascade_frontface.xml')
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

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

save_path = 'C:/Users/soyoo/Desktop/Final/realtime_version/dataset'

while(True):

    ret, img = cam.read()
    #img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        print(count)

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        #cv2.imwrite('dataset/' + str(count)+".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()