#importing libraries
import numpy as np
import pickle
import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

#human detector
import cv2 

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


#dog detector
from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.
from extract_bottleneck_features import *
import glob

dog_names = [item[20:-1] for item in sorted(glob.glob("dog_images/train/*/"))]

def VGG19_predict_breed(img_path):
    #load model
    VGG19_model = pickle.load(open("model.pkl", "rb"))
    # extract bottleneck features
    bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG19_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)].split('/')[-1].split('.')[-1].replace('_', ' ').lower()


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/', method = ['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
		    return redirect(request.url)
	    file = request.files['file']
	    if file.filename == '':
		    flash('No image selected for uploading')
		    return redirect(request.url)
	    if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join('uploads', filename))
            return redirect(url_for('result', filename = filename))
    return render_template('index.html')

@app.route('/result/<filename>')
def prediction(filename):
    face_det = face_detector(img_path)
    dog_det = dog_detector(img_path)
    
    if dog_det == True:
        result = 'I saw a ' + VGG19_predict_breed(img_path) + '. Don\'t get close to me.'
    
    if face_det == True:
        result = 'I found at least one human. Hooman, where is my fooood?'
    
    if dog_det == False and face_det == False:
        result = "I can't spot any hooman or doggo!!!!!"
    
    return render_template('result.html', result = result)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for(filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run()