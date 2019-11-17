'''
    File name: webcam_fashion_mnist.py
    Author: Steven Macías
    Date created: 17/11/2019
    Date last modified: 17/11/2019
    Python Version: 3.6
'''
import tensorflow as tf
from tensorflow import keras #API for tensor flow. Less code. High Level API. Avoid defining our own tensors
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import os.path
import random

threshold_value = 8
model_file_name = "model.h5"
key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
print("Number train images:\t"+str(len(train_images)))
print("Number test images:\t"+str(len(test_images)))

# Train the model
if(not os.path.exists(model_file_name)):
    #To make it easier to work with the model but matplot will show the same image.
    train_images = train_images/255.0
    test_images = test_images/255.0

    # Since we are working with 28x28 images we are going to have 784 input nodes
    # and since we have 10 labels we are going to have 10 output nodes.
    # We are going to have a hidden layer with 128 nodes.
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128, activation="relu"), # rectifier linear unit
        keras.layers.Dense(10, activation="softmax") # probability for each given class.
        ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_images, train_labels, epochs=10) #how many times your are going to see the sane image
    # Save model in a file
    model.save(model_file_name)
    # Evaluate accuracy
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Tested Acc: ", test_acc)
# Load the model
else:
    model = keras.models.load_model(model_file_name)

# Show how to the user how the input should be
prediction = model.predict(test_images)
random_val = random.randint(0, len(test_images))
plt.grid(False)
plt.imshow(test_images[random_val], cmap=plt.cm.binary)
plt.xlabel("Input example: "+class_names[test_labels[random_val]])
plt.title("Prediction: "+class_names[np.argmax(prediction[random_val])]) # Get the index with high probability
plt.show()

# Webcam loop
while True:
    try:
        check, frame = webcam.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Make the grey scale image have three channels
        grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
        (thresh, bw3c) = cv2.threshold(grey_3_channel, threshold_value, 255, cv2.THRESH_BINARY_INV)
        # Concatenate the frame and the black and white image
        img_concate_Hori=np.concatenate((frame,bw3c),axis=1)
        cv2.imshow("Capturing", img_concate_Hori)

        key = cv2.waitKey(1)
        if key == ord('s'):
            # Reduce the number of channels again
            img_ = cv2.cvtColor(bw3c, cv2.COLOR_RGB2GRAY)
            # Reduce the resolution
            img_ = cv2.resize(img_,(28,28))
            # Make it fit in our model
            img_ = img_/255.0
            prediction = model.predict([[img_]])
            # Plot results
            plt.grid(False)
            plt.imshow(img_, cmap=plt.cm.binary)
            plt.xlabel("Caputed image ")
            plt.title("Prediction: "+class_names[np.argmax(prediction[0])]) # Get the index with high probability
            plt.show()

        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

        elif key == ord('p'):
            threshold_value += 1
            print("threshold_value: ",threshold_value )

        elif key == ord('l'):
            threshold_value -= 1
            print("threshold_value: ",threshold_value )

    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break
