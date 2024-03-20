from model_convolucional import *
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
from torchvision import transforms
from PIL import ImageOps
import torch

model = SimpleCNN(num_classes=47)


class GUI(tk.Tk):
    
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.title("EMNIST Image Recognition")
        self.geometry("400x400")

        self.canvas = tk.Canvas(self, width=200, height=200)
        self.canvas.pack(pady=20)

        self.label_prediction = tk.Label(self, text="Prediction: ")
        self.label_prediction.pack()

        self.button_browse = tk.Button(self, text="Browse Image", command=self.browse_image)
        self.button_browse.pack()

    def browse_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            image = Image.open(file_path).convert("L")  # Convertir a escala de grises
            image = image.resize((28, 28))  # Resize to MNIST image size
            image = transforms.ToTensor()(image)
            image = transforms.Normalize((0.5,), (0.5,))(image)
            

        with torch.no_grad():
            output = self.model(image.unsqueeze(0)).cpu()
            _, predicted = torch.max(output.data, 1)

            self.display_image(file_path)
            self.label_prediction.config(text=f"Prediction: {chr(predicted.item() + 96)}")


# Crear una instancia de la GUI

    def display_image(self, file_path):
        image = Image.open(file_path)
        image = ImageTk.PhotoImage(image.resize((200, 200)))
        self.canvas.config(width=200, height=200)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
        self.canvas.image = image 


gui = GUI(model)
gui.mainloop()