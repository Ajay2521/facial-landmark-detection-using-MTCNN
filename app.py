#Import necessary libraries

# return a secure version of the user input file name
from werkzeug.utils import secure_filename

# Flask - Flask is an API of Python that allows us to build up web-applications.
# flash - used to generate informative messages in the flask
# request - used to gain access
# redirect - used to returns a response object and redirects the user to another target location
# url_for - used for creating a URL to prevent the overhead of having to change URLs throughout an application
# render_template - used to generate output from a template file based on the Jinja2 engine
# Response - container for the response data returned by application route functions, plus some additional information needed to create an HTTP response
from flask import Flask, flash, request, redirect, url_for, render_template, Response
from flask_ngrok import run_with_ngrok

# open cv for image processing
# “CVV” stands for “Card Verification Value” which is used to read, save and display the image
import cv2

# used for accessing the file and folder in the machine
import os

# VideoStream - used for video stream using webcam
from imutils.video import VideoStream

# imutils - used to make basic image processing functions such as translation, rotation, resizing, skeletonization, and displaying Matplotlib images
import imutils

# MTCNN - is a face detector which implements the paper Zhang, Kaipeng et al. 
# “Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks
from mtcnn import MTCNN

# matplotlib - used to visualize the image 
# import matplotlib.pyplot as plt


UPLOAD_FOLDER = './static/upload_image/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# creating a classifier for detecting the landmarks
classifier = MTCNN()
# print(classifier)
# prints the classifier object = <mtcnn.mtcnn.MTCNN object at 0x7fb8d59d2390>, value can be of random type

# function for locating the the facial landmark on static images
def landmark_image(imagePath):

  # reading the image
  image = cv2.imread(imagePath)
  # print(image) -> reading the image in the form of pixel value -> eg: [254 253 249]
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

    # face contains values like box, confidence, and keypoints for each iteration
    # print(faces)
    # print(faces['box']) # accessing the box value using key ="box" -> [164, 50, 49, 61] -> [x, y, w, h]

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
  cv2.imwrite("./static/upload_image/Landmarked_faces.png",image)



#Initialize the Flask app
app = Flask(__name__)
run_with_ngrok(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # setting up the upload folder to the app
app.secret_key = "secret-key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# function for checking the upload image extension
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# function for displaying the home page
@app.route('/')
def index():
    return render_template('index.html')

# function redirect to the upload page
@app.route('/upload')
def upload():
  return render_template('upload.html')

# once the upload & predict button has been click it invoke the following function
@app.route('/upload', methods=['POST'])
def upload_image():

	# checking for the presence of the file
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)

	# checking the uploaded file type is of allowed file type
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)

		# saving the file
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		flash('Image successfully uploaded and displayed below')
		return render_template('upload.html', filename = filename)

	else:
		flash('Allowed image types are png, jpg, jpeg, gif')
		return redirect(request.url)


# displaying the uploaded image
@app.route('/upload/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='upload_image/' + filename))

# displaying the predicted image
@app.route('/upload/predict/<filename>')
def predict_image_display(filename):
	imagePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
	landmark_image(imagePath)
	return redirect(url_for('static', filename='upload_image/Landmarked_faces.png' ))

# generating the frame from the webcam
def landmark_video():
	vs = VideoStream(src=0).start()

	while True:

		# read the frame from the threaded videostream and resize it and have width of 400
		_,frame = vs.read()
		# frame = imutils.resize(frame, width = 400)
		
		# detecting the no. of faces in a image
		faces = classifier.detect_faces(frame)
		
		# This detect_faces produce the output which is in json format(key:value pair/dictionary format) where it has
		# 'box': [164, 50, 49, 61] = coordinates of the bounding box which is x, y, w, h
		# 'confidence': 0.9996275901794434 = confidence -> likelihood on a population parameter, quantify the uncertainty on an estimate. 
		# 'keypoints': {'left_eye': (180, 71), 'right_eye': (203, 70), 'nose': (194, 84), 'mouth_left': (181, 97), 'mouth_right': (201, 97)}} 
		# keypoints = facial landmarks for the 5 important landmarks
		# The above details will be calculated for all the faces in the image
		
		# print(faces)
		
		# separate the bounding box coordinates
		for face in faces:
			
			# face contains values like box, confidence, and keypoints for each iteration
			# print(faces)
			
			# print(faces['box']) # accessing the box value using key ="box" -> [164, 50, 49, 61] -> [x, y, w, h]
			
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
				
		ret, buffer = cv2.imencode('.jpg',frame)
		frame = buffer.tobytes()
		yield (b'--frame\r\n'
		b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
	
		key = cv2.waitKey(1) & 0xFF
	
		if key == ord('q'):
			break
	vs.stop()	
	cv2.destoryAllWindows()

# redirect to the live page
@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/webcam')
def webcam():
    return Response(landmark_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

# running the flask app
if __name__ == "__main__":
    app.run()