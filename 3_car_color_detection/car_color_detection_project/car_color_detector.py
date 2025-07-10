import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Map RGB color to basic color names
def map_color_to_name(color):
    r, g, b = color
    if r > 200 and g > 200 and b > 200:
        return "White"
    elif r < 50 and g < 50 and b < 50:
        return "Black"
    elif r > 150 and g < 100 and b < 100:
        return "Red"
    elif r < 100 and g > 150 and b < 100:
        return "Green"
    elif r < 100 and g < 100 and b > 150:
        return "Blue"
    elif r > 150 and g > 150 and b < 100:
        return "Yellow"
    elif r > 150 and g < 100 and b > 150:
        return "Pink"
    elif r < 100 and g > 150 and b > 150:
        return "Cyan"
    elif r > 100 and g > 100 and b > 100:
        return "Gray"
    else:
        return "Unknown"

# Get dominant color using KMeans
def get_dominant_color(image, k=4):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img)
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]
    return dominant.astype(int)

# Main GUI function
def choose_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img_cv = cv2.imread(file_path)
        dominant_color = get_dominant_color(img_cv)
        color_name = map_color_to_name(dominant_color)

        # Convert image to show in GUI
        img_pil = Image.open(file_path)
        img_pil = img_pil.resize((300, 300))  # Resize for GUI
        img_tk = ImageTk.PhotoImage(img_pil)

        image_label.configure(image=img_tk)
        image_label.image = img_tk

        result_label.config(text=f"Detected Color: {color_name}\nRGB: {dominant_color}")

# GUI window
window = tk.Tk()
window.title("Car Color Detection")
window.geometry("400x450")

btn = tk.Button(window, text="Choose Car Image", command=choose_image)
btn.pack(pady=10)

image_label = tk.Label(window)
image_label.pack()

result_label = tk.Label(window, text="", font=("Arial", 14))
result_label.pack(pady=10)

window.mainloop()
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Replace these with your actual variables if named differently
# Example: y_test = actual labels, y_pred = predicted labels

print("✅ Model Evaluation Results:")
print("🔹 Accuracy:", accuracy_score(y_test, y_pred))
print("🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("🔹 Classification Report:\n", classification_report(y_test, y_pred))
print("y_test exists:", 'y_test' in locals())
print("y_pred exists:", 'y_pred' in locals())
