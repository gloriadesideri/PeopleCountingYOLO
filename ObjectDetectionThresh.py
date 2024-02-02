import cv2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from scipy import ndimage
from PIL import Image
import glob
import imutils

import os
class ObjectDetectionThresh:
    def __init__ (self, bg_path, datapath):
        self.datapath = datapath
        self.bg_path = bg_path
        self.images_normalized=[]
        self.binary_images=[]
        self.kernelized_images=[]
        self._normalize()
        self._remove_bg()
        self._binarize()
        self._kernelize()
        
    
    def _normalize(self):
        print("Normalizing images...")
        for filename in glob.glob(self.datapath+'/*.png'):
            #print("here")
            frame = cv2.imread(filename)

            # Normalize the image
            img_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            self.images_normalized.append(img_normalized)
    def _remove_bg(self):
        #self.images_normalized[48][0:100, 200:250]=0
        #self.images_normalized[48][400:512, 200:250]=0
        bg = cv2.imread(self.bg_path)
        for i in range(len(self.images_normalized)):
            self.images_normalized[i]= cv2.subtract(self.images_normalized[i], bg)
        self.images_normalized.pop(48)
    def _binarize(self):
        for i in range(len(self.images_normalized)):
            _, binary_image = cv2.threshold(self.images_normalized[i], 30, 255, cv2.THRESH_BINARY)
            self.binary_images.append(binary_image)
    def _kernelize(self):
        kernelSize= (5,5)
        for i in range(len(self.binary_images)):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
            closing = cv2.morphologyEx(self.binary_images[i], cv2.MORPH_CLOSE, kernel)
            self.kernelized_images.append(closing)
    def _show_image(self,image):
        cv2.imshow("image",image)
        cv2.waitKey(1)
        
    def detect(self):
        for i in range(len(self.kernelized_images)):
            depth_map = self.kernelized_images[i][:, :, 2]
            # Apply the binary mask to the original image to extract the filtered pixels
            contours, hierarchy  = cv2.findContours(depth_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_image = np.copy(self.kernelized_images[i])
            #contour_image = np.copy(images_normalized[i])
            filtered_contours = [contour for contour in contours if 300 < cv2.contourArea(contour) < 13000]
            for j, filtered_contour in enumerate(filtered_contours):
                area = cv2.contourArea(filtered_contour)
                print(f"Area of Object {j + 1} of image {i}: {area} square units")
            print(f"Number of people in image {i}: {j+1}")
            print("------------------------------------------------------------")

            cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 2)
            #self._show_image(contour_image)
            cv2.imshow('Object Detection', contour_image)

            # Break the loop if the 'q' key is pressed
            if cv2.waitKey(200) & 0xFF == ord('q'):
                break
    
        

            