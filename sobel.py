import numpy as np
import cv2
import argparse
from convolution import convolution
from gaussian_blur import gaussian_blur
import matplotlib.pyplot as plt
from tkinter import *
import tkinter as tk

class UI(Frame):
    def __init__(self, master, *args, **kwargs):
        Frame.__init__(self, master, *args, **kwargs)
        self.parent = master
        self.grid()
        self.createWidgets()

    def createWidgets(self):
        self.display = Label(self, font=("Arial", 13), borderwidth=0, justify="center", text="Introduce el nombre del archivo \nde imagen")
        self.display.grid(row=0, column=0, columnspan=1, sticky="nsew")

        userInput = Entry(self, font=("Arial", 18), borderwidth=0, justify="center")
        self.display = userInput
        self.display.grid(row=1, column=0, columnspan=1, sticky="nsew")

        self.ceButton = Button(self, font=("Arial", 12), fg='black', text="ocultamiento", command = lambda: start(userInput))
        self.ceButton.grid(row=2, column=0, sticky="nsew")

def sobel_edge_detection(image, filter, verbose=False):
    new_image_x = convolution(image, filter, verbose)

    if verbose:
        plt.imshow(new_image_x, cmap='gray')
        plt.title("Horizontal Edge")
        plt.show()

    new_image_y = convolution(image, np.flip(filter.T, axis=0), verbose)

    if verbose:
        plt.imshow(new_image_y, cmap='gray')
        plt.title("Vertical Edge")
        plt.show()

    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))

    if verbose:
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("Gradient Magnitude")
        plt.show()

    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    if verbose:
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("Gradient Magnitude")
        plt.show()

    return gradient_magnitude

if __name__ == '__main__':
    #sobel filter mask/kernel 
    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    #get image arguments from the shell  "python sobel.py -i image.jpg"
    ap = argparse.ArgumentParser()
    """ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())
    #use open cv 2 to change the image into an array of numbers 
    image = cv2.imread(args["image"]) 
    """
    ui = Tk()
    ui.title("Proyecto Semana TEC")
    ui.config(bg="#000000", cursor="cross", height="350", width="350", relief="groove")
    ui.resizable(False, False)
    root = UI(ui).grid()
    ui.mainloop()

    #blur the image using a gaussian filter
    image = gaussian_blur(image, 9, verbose=True)
    sobel_edge_detection(image, filter, verbose=True) 