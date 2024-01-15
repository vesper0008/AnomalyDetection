# Import necessary libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, classification_report, confusion_matrix
import os


def load_images(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            images.append(img.flatten())
            labels.append(label)
    return images, labels


defective_images, defective_labels = load_images("/Users/tolunkeles/PycharmProjects/Anomaly Detection/archive"
                                                 "/Defect_images", 1)
non_defective_images, non_defective_labels = load_images("/Users/tolunkeles/PycharmProjects/Anomaly Detection"
                                                         "/archive/NODefect_images/2306881-210020u", 0)

images = defective_images + non_defective_images
labels = defective_labels + non_defective_labels
images = np.array(images)
labels = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(images, labels, stratify=labels, test_size=0.2,
                                                    random_state=42)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)
plt.plot(fpr, tpr, label="AUC"+str(auc))
plt.ylabel('True Positive rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()
