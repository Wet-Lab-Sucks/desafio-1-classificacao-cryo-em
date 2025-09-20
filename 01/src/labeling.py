import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# GLOBAL CONFIG
IMAGE_DIR = 'dataset'
LABELS_FILE = 'labels.txt'
CLASSES = ['cube', 'sphere', 'torus', 'helix', 'dumbbell', 'star', 'ring_stack', 'random_blobs', 'layers', 'asymmetric']

# LABELING DISTRIBUTION 
# Jorge (0-4000)
#MIN = 0
#MAX = 4000 

# Felipe (4001-8000)
MIN = 4001
MAX = 8000 

# Isa (8001-12000)
#MIN = 8001
#MAX = 12000 

class ImageLabeler:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Labelling Tool")

        self.images = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')])[MIN:MAX]
        self.labels = self.load_labels()
        self.index = self.find_next_index()
        self.history = []  # <- image history

        # interface
        self.filename_label = tk.Label(master, text="", font=("Helvetica", 14))
        self.filename_label.grid(row=0, column=0, columnspan=2, pady=5)

        self.img_label = tk.Label(master)
        self.img_label.grid(row=1, column=0, rowspan=len(CLASSES) + 2, padx=10)

        self.feedback_label = tk.Label(master, text="", fg="green", font=("Helvetica", 12))
        self.feedback_label.grid(row=len(CLASSES)+2, column=0, columnspan=2, pady=5)

        for i, class_name in enumerate(CLASSES):
            button = tk.Button(master, text=class_name, width=15,
                               command=lambda c=class_name: self.label_image(c))
            button.grid(row=i+1, column=1, padx=10, pady=2)

        tk.Button(master, text="Pular", command=self.skip_image).grid(row=len(CLASSES)+1, column=1, pady=5)
        tk.Button(master, text="Voltar", command=self.prev_image).grid(row=len(CLASSES)+2, column=1, pady=5)

        self.show_image()

    def load_labels(self):
        labels = {}
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, 'r') as f:
                for line in f:
                    img, label = line.strip().split(',')
                    labels[img] = label
        return labels

    def save_labels_to_file(self):
        with open(LABELS_FILE, 'w') as f:
            for img, label in self.labels.items():
                f.write(f"{img},{label}\n")

    def show_image(self):
        if self.index >= len(self.images):
            messagebox.showinfo("Fim", "Todas as imagens foram classificadas!!!")
            self.master.quit()
            return

        img_name = self.images[self.index]
        img_path = os.path.join(IMAGE_DIR, img_name)

        # update labels
        self.labels = self.load_labels()

        pil_img = Image.open(img_path).resize((400, 400))
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.img_label.config(image=self.tk_img)

        self.filename_label.config(text=f"Imagem: {img_name}")
        current_label = self.labels.get(img_name, "Não classificada")
        self.feedback_label.config(text=f"Classificação: {current_label}",
                                fg="green" if current_label != "Não classificada" else "red")

    def label_image(self, label):
        img_name = self.images[self.index]
        self.labels[img_name] = label
        self.save_labels_to_file()
        self.feedback_label.config(text=f"Classificação salva: {label}", fg="green")

        self.history.append(self.index)
        self.index += 1
        self.show_image()

    def skip_image(self):
        self.feedback_label.config(text="Imagem pulada (sem classificação)", fg="orange")
        self.history.append(self.index)
        self.index += 1
        self.show_image()

    def prev_image(self):
        if not self.history:
            self.feedback_label.config(text="Não há imagem anterior para voltar", fg="gray")
            return

        self.index = self.history.pop()
        self.feedback_label.config(text=f"Voltando para {self.images[self.index]}", fg="blue")
        self.show_image()

    def find_next_index(self):
        for i, img in enumerate(self.images):
            if img not in self.labels:
                return i
        return len(self.images)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabeler(root)
    root.mainloop()
