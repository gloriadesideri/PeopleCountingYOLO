import ultralytics
ultralytics.checks()
import torch
import cv2
from ultralytics import YOLO
import os
from PIL import Image
import glob


class ObjectDetection:
  def __init__(self ,dataset='./',datapath='./runs/detect/train'):
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: ", self.device)
    self.model = self.load_model(datapath)
    self.datapath=datapath
    self.images_normalized= []
    self.dataset=dataset
    self._normalize()


  def _normalize(self):
      print("Normalizing images...")
      for filename in glob.glob(self.dataset+'/*.png'):
            #print("here")
          frame = cv2.imread(filename)

            # Normalize the image
          img_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
          self.images_normalized.append(img_normalized)
    
  

             
  def load_model(self, datapath):
    if os.path.exists(datapath):
      model = YOLO(datapath+'best.pt')
      return model
    else:
      print("Model not found")
      return None
    
    
  
  def predict(self, frame):
    results = self.model(frame)
    return results
    
  def plot_bboxes(self, results, frame):
    xyxys=[]
    confidences=[]
    class_ids=[]

    for result in results:
      boxes = result.boxes.cpu().numpy()
      xyxys.append(boxes.xyxy)
      confidences.append(boxes.conf)
      class_ids.append(boxes.cls)
    return results[0].plot, xyxys, confidences, class_ids
  
  
  def inference_from_frames(self):
    """inference from a folder that already contains every frame"""
    #image_files = [f for f in os.listdir(images_folder) if f.endswith('.png')]

    for image in self.images_normalized:
        # Construct the full path to the PNG image
        #image_path = os.path.join(images_folder, image_file)

        # Read the PNG image
        #frame = cv2.imread(image_path)

        # Make predictions using the ObjectDetection class
        results = self.predict(image)

        # Draw bounding boxes on the frame
        plot, xyxys, confidences, class_ids = self.plot_bboxes(results, image)
        frame_with_boxes = plot()

        # Display the frame with bounding boxes
        cv2.imshow('Object Detection', frame_with_boxes)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()
    
  def inference_from_video(self, video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        # Make predictions using the ObjectDetection class
        results = self.predict(frame)

        # Draw bounding boxes on the frame
        plot, xyxys, confidences, class_ids = self.plot_bboxes(results, frame)
        frame_with_boxes = plot()

        # Display the frame with bounding boxes
        cv2.imshow('Object Detection', frame_with_boxes)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()