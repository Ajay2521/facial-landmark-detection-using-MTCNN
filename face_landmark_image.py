
# Importing the required libraries

# MTCNN - is a face detector which implements the paper Zhang, Kaipeng et al. 
# “Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks.”
import mtcnn
from mtcnn import MTCNN

# “CVV” stands for “Card Verification Value” which is used to read, save and display the image
import cv2

# matplotlib - used to visualize the image 
# import matplotlib.pyplot as plt

# creating a classifier for detecting the landmarks
classifier = MTCNN()
print(classifier) # prints the classifier object = <mtcnn.mtcnn.MTCNN object at 0x7fb8d59d2390>, value can beof random type

def landmark_image(imagePath):

  # reading the image
  image = cv2.imread(imagePath)
  # print(image) -> reading the image in the form of pixcel value -> eg: [254 253 249]
  # plt.imshow(image) -> prints image in the form of BGR format
  
  # detecting the no. of faces in a image
  faces = classifier.detect_faces(image)
  
  # This detect_faces produce the output which is in json format(key:value pair/dictionary format) where it has
  # 'box': [164, 50, 49, 61] = coordinates of the bounding box which is x, y, w, h
  # 'confidence': 0.9996275901794434 = confidence -> likelihood on a population parameter, quantify the uncertainty on an estimate. 
  # 'keypoints': {'left_eye': (180, 71), 'right_eye': (203, 70), 'nose': (194, 84), 'mouth_left': (181, 97), 'mouth_right': (201, 97)}} 
  # keypoints = facial landmarks for the 5 important landmarks
  # The above details will be calculated for all the faces in the image
  
  # print(faces)  

  # separate the bounding box coordinates 
  for face in faces:
  
    # face contains values like box, confidece, and keypoints for each iteration
    # print(faces) 
    # print(faces['box']) # accesing the box value using key ="box" -> [164, 50, 49, 61] -> [x, y, w, h]

    # seperating the box value according to the x - axis, y - axis, w - width, h - height
    x,y,w,h = face['box']
    # print(x) -> x axis -> 164
    # print(y) -> y axis -> 50
    # print(w) -> width -> 49
    # print(h) -> height -> 61

    # separate the facial landmark coordinates 
    for key, value in face['keypoints'].items():
    
      # make a circle marks on the keypoints which represent the facial landmarks
      cv2.circle(image, value, 3, (0,255,0))
    
  # Show the image
  cv2.imwrite("Landmarked_faces.jpg",image)

  
