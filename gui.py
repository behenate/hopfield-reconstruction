import tkinter as tk
from tkinter import ttk
from tkinter import *

import numpy as np
from PIL import ImageTk, Image
from os import listdir
from os.path import isfile, join

from PIL.Image import Resampling
from hopfield_clouds import HopfieldClouds


# root.columnconfigure(0, weight=1)
# root.columnconfigure(1, weight=3)


class GUI:
    def __init__(self):
        self.picture_size = 420
        self.network = HopfieldClouds(130 ** 2)
        self.root = tk.Tk()
        self.root.geometry('1280x500')
        self.root.title('Hopfield Clouds')

        self.next_button = ttk.Button(self.root, text='>', command=self.next_image)
        self.next_button.grid(row=1, column=0, sticky=tk.E)

        self.prev_button = ttk.Button(self.root, text='<', command=self.prev_image)
        self.prev_button.grid(row=1, column=0, sticky=tk.W)

        self.original_img = self.network.get_current_image()
        self.original_img = self.original_img.resize((self.picture_size, self.picture_size), Resampling.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(self.original_img)

        original_frame = Frame(self.root, width=self.picture_size, height=self.picture_size)
        original_frame.grid(row=0, columnspan=1, sticky='we')
        self.original_image_label = Label(original_frame, image=self.img_tk)
        self.original_image_label.grid(row=1, column=0)

        self.cropped_img = Image.fromarray(np.uint8(np.zeros((self.picture_size, self.picture_size, 3))))
        self.cropped_img = ImageTk.PhotoImage(self.cropped_img)
        self.cropped_frame = Frame(self.root, width=self.picture_size, height=self.picture_size)
        self.cropped_frame.grid(row=0, column=1, sticky='we')
        self.cropped_image_label = Label(self.cropped_frame, image=self.cropped_img)
        self.cropped_image_label.grid(row=1, column=1)

        self.current_value = tk.DoubleVar()
        #  slider
        self.slider = ttk.Scale(self.root, from_=1, to=99, orient='horizontal', command=self.slider_changed,
                                variable=self.current_value)
        self.slider.set(50)
        self.slider.bind('<ButtonRelease-1>', self.slider_up)
        self.slider_label = Label(self.root, text='Percentage to crop:')
        self.slider_label.grid(row=1, column=1, columnspan=1, sticky='we')
        self.slider.grid(column=1, columnspan=1, row=2, sticky='we')

        self.value_label = ttk.Label(self.root, text=self.get_current_value())
        self.value_label.grid(row=3, column=1, columnspan=1, sticky='n')

        self.reconstructed_img = Image.fromarray(np.uint8(np.zeros((self.picture_size, self.picture_size, 3))))
        self.reconstructed_img = ImageTk.PhotoImage(self.reconstructed_img)
        self.reconstructed_frame = Frame(self.root, width=self.picture_size, height=self.picture_size)
        self.reconstructed_frame.grid(row=0, column=2, columnspan=1, sticky='n')
        self.reconstructed_image_label = Label(self.reconstructed_frame, image=self.reconstructed_img)
        self.reconstructed_image_label.grid(row=1, column=2, columnspan=1)
        self.reconstruct_button = ttk.Button(self.root, text='Reconstruct', command=self.reconstruct)
        self.reconstruct_button.grid(row=1, column=2, sticky='n')

        self.slider_up(None)
        self.root.mainloop()



    def slider_changed(self, event):
        self.value_label.configure(text=self.get_current_value())

    def get_current_value(self):
        return '{: .2f}'.format(self.current_value.get())

    def next_image(self):
        img = self.network.next_image()
        self.original_img = img.resize((self.picture_size, self.picture_size), Resampling.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(self.original_img)
        self.original_image_label.configure(image=self.img_tk)
        self.slider_up(None)

    def prev_image(self):
        img = self.network.prev_image()
        self.original_img = img.resize((self.picture_size,self.picture_size), Resampling.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(self.original_img)
        self.original_image_label.configure(image=self.img_tk)
        self.slider_up(None)


    def reconstruct(self):
        cropped, reconstructed = self.network.get_current_image_predictions(int(self.current_value.get()))
        self.reconstructed_img = reconstructed
        self.reconstructed_img = self.reconstructed_img.resize((self.picture_size, self.picture_size), Resampling.LANCZOS)
        self.reconstructed_img = ImageTk.PhotoImage(self.reconstructed_img)
        self.reconstructed_image_label.configure(image=self.reconstructed_img)

    def slider_up(self, event):
        cropped = self.network.get_current_cropped(int(self.current_value.get()))
        self.cropped_img = cropped
        self.cropped_img = self.cropped_img.resize((self.picture_size, self.picture_size), Resampling.LANCZOS)
        self.cropped_img = ImageTk.PhotoImage(self.cropped_img)
        self.cropped_image_label.configure(image=self.cropped_img)


gui = GUI()
