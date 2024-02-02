from ObjectDetectionThresh import ObjectDetectionThresh
import cv2
import matplotlib.pyplot as plt

#main
if __name__ == "__main__":
    datapath='./depths'
    detector=ObjectDetectionThresh('./bg.png',datapath)
    detector.detect()