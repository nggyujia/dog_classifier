# Dog Breed Classifier

## Overview
This is the capston project of the Udacity Data Science nanodegree program. The project experiements with different convolutional neural network (CNN) to predict dog breed based on image input. The final model uses a VGG19 model the keras package with added layers. A human face detector and a dog face detector are also created to first identify whether there is face present in an image, and if so, the model will proceed to predict the most similar dog breed to the human face or the dog breed for the dog face. The model is implemented in a web app. 

### Running the App
1. Go over the `classifier.ipynb` in the project's root directory to set up the data and model.

2. Run the following command in the app's directory to run your web app.
    `python app.py`

3. Go to http://0.0.0.0:3001/

### Using the App
Upload an image and the model will return a result.

## Analysis
The image inputted goes through processing steps that standardize the image into a format that can feed into the detector and the CNN model. Images are split into train and test set to evaluate model performance. Different layers and parameters are added and adjusted to test model preformance as well. More details are in `Dog Breed Classifier - files\dog_app.ipynb`.

## Results
The model is tested on 5 images found on the internet. It is able to correctly detect faces in 4/5 of the images and predict correctly the dog breeds. 

## Conclusion
This project provides an overview of image classification and the usage of CNN in this task. An ongoing challenge, image classification certainly attracts many interested developers as well as users to identify not only faces but also plants and many other objects. One particular challenge during the project was storage and running time, since the data are images, which take up a big space. Other keras pre-trained models can be experimented and different layers can be added as well as adjusting parameters to see if model performance could increase. 
