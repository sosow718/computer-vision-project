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
from demo import demo
from model import train_model, valid_model
import tensorflow as tf
import cv2
import numpy as np
import sys
import tensorflow as tf
from PIL import Image
from model import predict, image_to_tensor, deepnn

CASC_PATH = './data/haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
flags =  tf.app.flags
flags.DEFINE_string('MODE', 'demo',
                    'Set program to run in different mode, include train, valid and demo.')
flags.DEFINE_string('checkpoint_dir', './ckpt',
                    'Path to model file.')
flags.DEFINE_string('train_data', './data/fer2013/fer2013.csv',
                    'Path to training data.')
flags.DEFINE_string('valid_data', './valid_sets/',
                    'Path to training data.')
flags.DEFINE_boolean('show_box', False,
                    'If true, the results will show detection box')
FLAGS = flags.FLAGS

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = 'C:/Users/soyoo/Desktop/Final/image_version/haarcascade_frontface.xml'
faceCascade = cv2.CascadeClassifier(cascadePath);

def format_image(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = cascade_classifier.detectMultiScale(
    image,
    scaleFactor = 1.3,
    minNeighbors = 5
  )
  # None is no face found in image
  if not len(faces) > 0:
    return None, None
  max_are_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
      max_are_face = face
  # face to image
  face_coor =  max_are_face
  image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
  # Resize image to network size
  try:
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
  except Exception:
    print("[+} Problem during resize")
    return None, None
  return  image, face_coor
# iniciate id counter
id = 0
names = []
db = open("database.txt", "r")
f1 = db.readlines()

for x in f1:
    names.append(x[0:-1])
db.close()

face_x = tf.placeholder(tf.float32, [None, 2304])
y_conv = deepnn(face_x)
probs = tf.nn.softmax(y_conv)

saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
sess = tf.Session()
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

feelings_faces = []
for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(cv2.imread('./data/emojis/' + emotion + '.png', -1))
emoji_face = []

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

        cv2.putText(img, str(id), (x + 5, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1)

    detected_face, face_coor = format_image(img)
    result = None
    if detected_face is not None:
        tensor = image_to_tensor(detected_face)
        result = sess.run(probs, feed_dict={face_x: tensor})
    if result is not None:
      for index, emotion in enumerate(EMOTIONS):
        cv2.putText(img, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
        cv2.rectangle(img, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4),
                      (255, 0, 0), -1)
        emoji_face = feelings_faces[np.argmax(result[0])]
      for c in range(0, 3):
        img[200:320, 10:130, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + img[200:320, 10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
    else:
      comment = 'We cannot detect a face....Too bad!'
      cv2.putText(img, comment, (115, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    cv2.imwrite('result of ' +str(id) + '.jpg', img)
    # cv2.namedWindow('Guess who I am and how I feel', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Guess who I am and how I feel', 450, 600)
    cv2.imshow('Guess who I am and how I feel', img)




    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cv2.destroyAllWindows()