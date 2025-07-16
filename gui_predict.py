import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained model
model = load_model('dog_cat_cnn.h5')

# Define image size same as during training
img_size = (64, 64)

def classify_image(file_path):
    img = image.load_img(file_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    result = model.predict(img_array)
    class_index = np.argmax(result)
    if class_index == 0:
        return "Cat"
    else:
        return "Dog"

def upload_and_predict():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk

        label_result.config(text="Predicting...")
        prediction = classify_image(file_path)
        label_result.config(text=f"Prediction: {prediction}")

# Create GUI window
root = tk.Tk()
root.title("Cat & Dog Classifier")

btn_upload = tk.Button(root, text="Upload Image", command=upload_and_predict)
btn_upload.pack()

panel = tk.Label(root)
panel.pack()

label_result = tk.Label(root, text="", font=("Arial", 14))
label_result.pack()

root.mainloop()

