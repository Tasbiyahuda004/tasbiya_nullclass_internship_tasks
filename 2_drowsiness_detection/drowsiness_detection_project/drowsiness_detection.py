import cv2
from tkinter import *
from tkinter import messagebox
import threading

# Load Haarcascade files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Flag to track drowsiness
drowsy = False

def detect_drowsiness():
    global drowsy
    cap = cv2.VideoCapture(0)
    closed_eyes_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) == 0:
                closed_eyes_frames += 1
            else:
                closed_eyes_frames = 0

            # Show red warning after 15 frames with no eyes
            if closed_eyes_frames >= 15:
                drowsy = True
                cv2.putText(frame, "DROWSINESS DETECTED!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                drowsy = False

        cv2.imshow('Drowsiness Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def start_detection():
    messagebox.showinfo("Starting", "Drowsiness detection has started.\nPress 'q' to stop.")
    threading.Thread(target=detect_drowsiness).start()

# GUI using Tkinter
root = Tk()
root.title("Drowsiness Detection System")
root.geometry("400x200")
root.config(bg="#e0f7fa")

label = Label(root, text="Drowsiness Detection System", font=("Arial", 18, "bold"), bg="#e0f7fa")
label.pack(pady=20)

start_button = Button(root, text="Start Detection", command=start_detection, font=("Arial", 14), bg="#00796b", fg="white")
start_button.pack(pady=10)

root.mainloop()
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Replace these with your actual variables if named differently
# Example: y_test = actual labels, y_pred = predicted labels

print("✅ Model Evaluation Results:")
print("🔹 Accuracy:", accuracy_score(y_test, y_pred))
print("🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("🔹 Classification Report:\n", classification_report(y_test, y_pred))
print("y_test exists:", 'y_test' in locals())
print("y_pred exists:", 'y_pred' in locals())
