from ObjectDetectionYOLO import ObjectDetection
import cv2
import matplotlib.pyplot as plt

#main
if __name__ == "__main__":
    datapath='./'
    detector=ObjectDetection(dataset='./depths',datapath=datapath)
    detector.inference_from_frames()