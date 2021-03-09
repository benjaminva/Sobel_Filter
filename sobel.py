import numpy as np
import cv2
import argparse
from convolution import convolution

if __name__ == '__main__':
    #sobel filter mask/kernel 
    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    #get image arguments from the shell  "python sobel.py -i image.jpg"
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())
    #use open cv 2 to change the image into an array of numbers 
    image = cv2.imread(args["image"]) 

    # To test the first part of convolution function
    convolution(image, filter, verbose=True)