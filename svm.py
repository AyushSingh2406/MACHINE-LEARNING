import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("Loading dataset...")
train_data = pd.read_csv("mnist_train.csv")     
test_data = pd.read_csv("mnist_test.csv")  
print("Dataset loaded successfully!")
print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)
print(train_data.head())
print(test_data.head())
X_train = train_data.drop(columns=["label"]).values  
y_train = train_data["label"].values  
X_test = test_data.drop(columns=["label"]).values  
y_test = test_data["label"].values  
fig, axes = plt.subplots(1, 5, figsize=(10, 5))
for i, ax in enumerate(axes):
    ax.imshow(X_train[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis("off")
plt.show()
X_train_small = X_train[:10000]
y_train_small = y_train[:10000]
print("Training SVM classifier...")
svm_clf = SVC(kernel='linear')  
svm_clf.fit(X_train_small, y_train_small)
print("Training completed!")
print("Making predictions...")
y_pred = svm_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)