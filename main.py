import cv2
import tensorflow as tf
import numpy as np
# import skimage as ski

# determine video capture device
cam = cv2.VideoCapture(0)
cam.set(3, 224)
cam.set(4, 224)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path='./tflite_models/model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# # USES SKIMAGE
# test_img = ski.io.imread('D:/Coding/KelasAI/face_recog/unknown/test1.jpg')
# resized_img = ski.transform.resize(test_img, (224, 224)).astype('float32')

# # USES OPENCV
# test_img = cv2.imread('D:/Coding/KelasAI/face_recog/test-models/Frans Joddy/image.jpg')
# test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
# resized_img = cv2.resize(test_img, (224, 224))
# resized_img = resized_img.astype(np.float32)

# # Test the TFLite model on input data.
# resized_img = np.expand_dims(resized_img, axis = 0)
# interpreter.set_tensor(input_details['index'], resized_img)
# interpreter.invoke()
# output_data = interpreter.get_tensor(output_details['index'])
# print(float(output_data.argmax()))

while True:
    ret, img = cam.read()
    resized_img = cv2.resize(img, (224, 224))
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img.astype(np.float32)

    # Test the TFLite model on input data.
    gray_img = np.expand_dims(gray_img, axis = 0)
    gray_img = np.expand_dims(gray_img, axis = 0)
    interpreter.set_tensor(input_details['index'], gray_img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])
    print('score: ' + str(float(output_data.argmax())))

    cv2.imshow('Face Detection', img)
    # esc to exit
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()