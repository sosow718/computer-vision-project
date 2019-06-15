from google_images_download import google_images_download  # importing the library
import os
import cv2
from PIL import Image

CASC_PATH = './data/haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(image, scaleFactor = 1.3, minNeighbors = 5)
  # None is no face found in image
    if not len(faces) > 0:
        return None, None
    max_are_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
            max_are_face = face
    # face to image
    face_coor =  max_are_face
    return  face_coor


def downloadFiles(name):
    response = google_images_download.googleimagesdownload()  # class instantiation
    arguments = {"keywords": name, "limit": 50, "print_urls": True,
             'usage_rights': "labeled-for-reuse-with-modifications", "format": 'jpg'}  # creating list of arguments
    paths = response.download(arguments)  # passing the arguments to the function

def changeName(name):
    i = 1
    os.mkdir(name + '/')
    for filename in os.listdir("downloads/" + name + "/"):
        dst = name + str(i) + ".jpg"
        src = "downloads/" + name + "/" + filename
        dst = name + '/' + dst
        os.rename(src, dst)
        i += 1


def faceCheck(name):
    for filename in os.listdir("downloads/" + name + "/"):
        print(filename)
        frame = cv2.imread("downloads/" + name + "/" + filename)
        face_coor = format_image(frame)
        if face_coor is None:
            os.remove("downloads/" + name + "/" + filename + ".jpg")


def main():
    name = input("Enter the name of the celebrity: ")
    downloadFiles(name)
    print('Checking for photos without faces...')
    faceCheck(name)
    print('Removed photos without faces.')
    print('and now changing file names...')
    changeName(name)
    print('done!')

main()