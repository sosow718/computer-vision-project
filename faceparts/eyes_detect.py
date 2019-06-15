import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

eye_detector = cv2.CascadeClassifier('xmlFiles/Eyes.xml')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")

while(True):
    ret, img = cam.read()
    cv2.putText(img, "Eyes detection\n Press Q", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    eye = eye_detector.detectMultiScale(img, 1.3, 5)

    for (x,y,w,h) in eye:
        cv2.rectangle(img, (x,y), (x+h,y+w), (0,0,255), 2)
        cv2.imwrite("feature/Eyes.jpg", img[y:y+w,x:x+h])
        cv2.imshow('image', img)

    if cv2.waitKey(1) == ord('q'): break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()