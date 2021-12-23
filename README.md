# face-detect-cam
Use a webcam to detect faces within a dataset.

Made with OpenCV, TensorFlow, and TensorFlow Lite model.

## Dependencies
Made with Python 3.9.7
A bit of GUI is made with Tkinter and OpenCV.
```
?=Generated by pipreqs=?
numpy==1.21.5
opencv_python==4.5.4.60
tensorflow==2.7.0
tflite_model_maker==0.3.4
```
Everything was installed using pip.

## What are those files?
### get_face_data.py
Uses Tkinter, OpenCV, time, and os modules.
- Makes a new folder to store data (photo of a face) inside folder "test-models".
- Captures/snapshots a frame from a video input to take a photo.
- Photos are stored in 224x224 resolution.

_SUBJECT TO CHANGE_

### tflite_model_train.py
Uses tflite_model_maker module.
- Makes a TensorFlow Lite model using the available dataset in folder "test-models".
- Model was made using float32 format.

_SUBJECT TO CHANGE_

### main.py
_Unfinished_
- Get video input from a camera.
- Compares faces that appear in the camera with the created tflite model.

## Developer
Albert E (vradnisntlong@gmail.com)
