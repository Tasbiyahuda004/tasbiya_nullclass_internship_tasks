import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# Load and preprocess the image
img_path = 'sample.jpg'  # This should be in your folderc
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Get feature maps
features = model.predict(x)

# Visualize first 9 feature maps
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(features[0, :, :, i], cmap='viridis')
    plt.axis('off')
plt.tight_layout()
plt.show()
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Replace these with your actual variables if named differently
# Example: y_test = actual labels, y_pred = predicted labels

print("✅ Model Evaluation Results:")
print("🔹 Accuracy:", accuracy_score(y_test, y_pred))
print("🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("🔹 Classification Report:\n", classification_report(y_test, y_pred))
print("y_test exists:", 'y_test' in locals())
print("y_pred exists:", 'y_pred' in locals())
