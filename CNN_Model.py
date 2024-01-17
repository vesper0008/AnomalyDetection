import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import os


def load_images(folder_path, label):
    images_list = []
    labels_list = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            images_list.append(img.flatten())
            labels_list.append(label)
    return images_list, labels_list


defective_images, defective_labels = load_images("archive/Defect_images", 1)
non_defective_images, non_defective_labels = load_images("archive/NODefect_images/2306881-210020u", 0)
images = np.array(defective_images + non_defective_images)
labels = np.array(defective_labels + non_defective_labels)
images = images / 255.0
X_train, X_test, y_train, y_test = train_test_split(images, labels, stratify=labels, test_size=0.33,
                                                    random_state=42)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train.reshape(-1, 224, 224, 1), y_train, epochs=50, validation_data=(X_test.reshape(-1, 224, 224, 1), y_test))
test_loss, test_acc = model.evaluate(X_test.reshape(-1, 224, 224, 1), y_test, verbose=2)
print('\nTest accuracy:', test_acc)
