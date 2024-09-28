import os
import glob
import numpy as np
import random
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Initial parameters
epochs = 100
lr = 1e-3
batch_size = 64
img_dims = (96, 96, 3)

data = []
labels = []

# Load image files from the dataset
dataset_path = r'Wivsafe-Detection\Violence-Detector'
violent_images = glob.glob(os.path.join(dataset_path, 'violent', '*.jpg'))
non_violent_images = glob.glob(os.path.join(dataset_path, 'non-violent', '*.jpg'))

# Combine images and labels
for img in violent_images:
    image = cv2.imread(img)
    if image is None:
        print(f"Could not load image: {img}")
        continue  # Skip if the image could not be loaded
    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    image = img_to_array(image)
    data.append(image)
    labels.append(1)  # Label for violent

for img in non_violent_images:
    image = cv2.imread(img)
    if image is None:
        print(f"Could not load image: {img}")
        continue  # Skip if the image could not be loaded
    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    image = img_to_array(image)
    data.append(image)
    labels.append(0)  # Label for non-violent

# Pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split dataset for training and validation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(img_dims[0], img_dims[1], img_dims[2])))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation("softmax"))

# Compile the model
opt = Adam(learning_rate=lr)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=epochs, batch_size=batch_size, verbose=1)

# Save the model to disk
model.save(r'C:\Users\junai\Desktop\Women in danger data set\Violence-Detector\violence_detection.keras')
 # Save in .keras format

# Plot training/validation loss/accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# Save plot to disk
plt.savefig(r'C:\Users\junai\Desktop\Women in danger data set\Violence-Detector\plot.png')

