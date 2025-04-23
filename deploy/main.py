import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# Restrict TensorFlow GPU memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load the model
model = load_model("D:/Keras/Sign_Language_Classcification/model/Sign_Language_recognition_model.h5")

# Preprocess image to feed into model
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 224, 224, 3)
    return image_array

# Predict function
def predict_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    image = Image.open(file_path)
    display_image = image.resize((200, 200))
    img = ImageTk.PhotoImage(display_image)
    image_label.config(image=img)
    image_label.image = img

    processed = preprocess_image(file_path)
    prediction = model.predict(processed)
    predicted_class = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                       'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                       'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                       'U', 'V', 'W', 'X', 'Y', 'Z']
    predicted_index = np.argmax(prediction)
    predicted_label = predicted_class[predicted_index]
    result_label.config(text=f"Predicted: {predicted_label}")

# GUI setup
root = tk.Tk()
root.title("Sign Language Classifier")
root.geometry("300x350")

tk.Button(root, text="Upload Image", command=predict_image).pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="Prediction will appear here", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
