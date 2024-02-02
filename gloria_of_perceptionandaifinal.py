import cv2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from scipy import ndimage
from PIL import Image
import glob
import imutils

import os

# Load ToF image
images_normalized=[]

#Gloria path
#for filename in glob.glob('/content/drive/MyDrive/PerceptionAI/data/depths/*.png'):
i=0;
#Julie Path
for filename in glob.glob('/content/drive/MyDrive/PerceptionAI/data/depths/*.png'):
    #print("here")
    frame = cv2.imread(filename)

    # Normalize the image
    img_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    images_normalized.append(img_normalized)
    # Display the normalized image
    print(i);
    cv2_imshow(img_normalized)
    cv2.waitKey(0)

    # Obtenez le nom du fichier sans le chemin
    file_name = os.path.basename(filename)

    # Construisez le chemin du fichier de sortie
    output_path = os.path.join('/content/drive/MyDrive/PerceptionAI/data/normalized', file_name)

    # Enregistrez l'image normalisée
    cv2.imwrite(output_path, img_normalized)
    i=i+1

# Close all windows
#cv2.destroyAllWindows()

"""#### Thresholding methods
The initial idea was to extract only the values between 254 and 170 but my code has been deleted
"""

cv2_imshow(images_normalized[36])

# Crop the image
cropped_image = images_normalized[36][400:512, 200:250]
cv2_imshow(cropped_image)

images_normalized[36][0:100, 200:250]=0
images_normalized[36][400:512, 200:250]=0
cv2_imshow(images_normalized[36])

#bg
bg= images_normalized[36]

result=cv2.subtract(images_normalized[1], bg)
cv2_imshow(result)

images_normalized.pop(36)

for i in range(len(images_normalized)):
  images_normalized[i]= cv2.subtract(images_normalized[i], bg)
for image_n in images_normalized:
  cv2_imshow(image_n)
  cv2.waitKey(0)

#findcontours
#connected component labeling
images_normalized[0].shape

cv2_imshow(images_normalized[0])

#first method I try is to retrieve with a single filter knowing that the camera is positioned at 255 centimeters from the ground I get everithing between 170 and 254
depth_map = images_normalized[1][:, :, 2]

# Set the lower and upper thresholds for filtering
lower_threshold = 30
upper_threshold = 180

# Create a binary mask based on the threshold values
binary_mask = cv2.inRange(depth_map, lower_threshold, upper_threshold)

# Apply the binary mask to the original image to extract the filtered pixels
filtered_image = cv2.bitwise_and(depth_map, depth_map, mask=binary_mask)

# Display the original and filtered images
#cv2_imshow(depth_map)
cv2_imshow( filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#good but I still get the wall
contours, hierarchy  = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = np.copy(images_normalized[0])
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    print(f"Area of Object {i + 1}: {area} square units")

# Draw contours on the original image
contour_image = np.copy(images_normalized[1])
filtered_contours = [contour for contour in contours if 3000 < cv2.contourArea(contour) < 9000]

cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 2)
cv2_imshow(contour_image)

def show_image(image):
    cv2_imshow(image)
    c = cv2.waitKey()
    if c >= 0 : return -1
    return 0

for i in range(len(images_normalized)):
  depth_map = images_normalized[i][:, :, 2]
  # Apply the binary mask to the original image to extract the filtered pixels
  filtered_image = cv2.bitwise_and(depth_map, depth_map, mask=binary_mask)
  contours, hierarchy  = cv2.findContours(depth_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contour_image = np.copy(images_normalized[i])
  #contour_image = np.copy(images_normalized[i])
  filtered_contours = [contour for contour in contours if 600 < cv2.contourArea(contour) < 13000]
  for j, filtered_contour in enumerate(filtered_contours):
    area = cv2.contourArea(filtered_contour)
    print(f"Area of Object {j + 1} of image {i}: {area} square units")
  print("------------------------------------------------------------")

  cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 2)
  show_image(contour_image)

threshold_value = 30

# Apply thresholding to create a binary image
_, binary_image = cv2.threshold(images_normalized[0], threshold_value, 255, cv2.THRESH_BINARY)
cv2_imshow(binary_image)

binary_images=[]
for i in range(len(images_normalized)):
  _, binary_image = cv2.threshold(images_normalized[i], threshold_value, 255, cv2.THRESH_BINARY)
  cv2_imshow(binary_image)
  binary_images.append(binary_image)

# loop over the kernels sizes again
kernelSizes = [(5, 5)]
for kernelSize in kernelSizes:
	# construct a rectangular kernel form the current size, but this
	# time apply a "closing" operation
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
	closing = cv2.morphologyEx(binary_images[1], cv2.MORPH_CLOSE, kernel)
	cv2_imshow( closing)
	cv2.waitKey(0)

kernelSize= (5,5)
kernelized_images=[]
for i in range(len(binary_images)):
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
  closing = cv2.morphologyEx(binary_images[i], cv2.MORPH_CLOSE, kernel)
  kernelized_images.append(closing)
  cv2_imshow( closing)
  cv2.waitKey(0)

for i in range(len(kernelized_images)):
  depth_map = kernelized_images[i][:, :, 2]
  # Apply the binary mask to the original image to extract the filtered pixels
  filtered_image = cv2.bitwise_and(depth_map, depth_map, mask=binary_mask)
  contours, hierarchy  = cv2.findContours(depth_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contour_image = np.copy(kernelized_images[i])
  #contour_image = np.copy(images_normalized[i])
  filtered_contours = [contour for contour in contours if 600 < cv2.contourArea(contour) < 13000]
  for j, filtered_contour in enumerate(filtered_contours):
    area = cv2.contourArea(filtered_contour)
    print(f"Area of Object {j + 1} of image {i}: {area} square units")
  print("------------------------------------------------------------")

  cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 2)
  show_image(contour_image)

"""#### EXPORT THE DATASET"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install ultralytics
import ultralytics
ultralytics.checks()

from ultralytics import YOLO

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="NxbQbD7vMDita8gMWhn4")
project = rf.workspace("perception-ia").project("project-ltc1m")
dataset = project.version(2).download("yolov8")

model = YOLO('yolov8n.pt')

import datetime

results = model.train (data='/content/project-2/data.yaml', epochs = 300,imgsz=640)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir /content/runs

#EXPORT
model.export()
# safe the model
#!cp '/content/runs/detect/train8/weights/best.torchscript' '/content/drive/MyDrive/SE-5104A/tp'

"""#### TEST DU MODELE"""

from google.colab import drive
drive.mount('/content/drive')

import cv2
import numpy as np
from google.colab.patches import cv2_imshow

model = YOLO("/content/drive/MyDrive/best_2.pt")

"""Normalisation du dataset TEST

Test du modèle sur les images normalisées
"""

import glob

# Load ToF image
images_normalized = []

# Julie Path
for filename in glob.glob('/content/drive/MyDrive/PerceptionAI/data/test_1/*.png'):
    # Load and normalize the image
    frame = cv2.imread(filename)
    img_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    images_normalized.append(img_normalized)

# Perform prediction on each normalized image
for img_normalized in images_normalized:
    # Placeholder for the actual prediction logic, adjust this according to your YOLO model's usage
    results = results.predict(img_normalized, conf=0.5)

    # Display the normalized image
    plt.imshow(cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB))
    plt.show()

    # Display the results of the prediction (if there are any)
    if len(results) > 0:
        for r in results:
            res_plotted = results[0].plot()
            box = r.boxes.cpu()
            box_sz = box.xyxy.numpy()
            if box_sz.size != 0:
                plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
                plt.show()
                break

"""Autres essais"""

import os

# Chemin du répertoire contenant les images
images_directory = '/content/drive/MyDrive/PerceptionAI/data/test_1/'

# Liste de tous les fichiers dans le répertoire
image_files = [f for f in os.listdir(images_directory) if os.path.isfile(os.path.join(images_directory, f))]

# Tri des fichiers pour garantir l'ordre d'acquisition
image_files.sort()

# On lit chaque image du répertoire
for image_file in image_files:
    image_path = os.path.join(images_directory, image_file)
    frame = cv2.imread(image_path)

    # Vous pouvez ajouter ici le changement de taille de l'image si nécessaire

    results = model.predict(frame, conf=0.5)  # prédiction sur une image

    if len(results) > 0:
        for r in results:
            res_plotted = results[0].plot()
            box = r.boxes.cpu()
            box_sz = box.xyxy.numpy()
            if box_sz.size != 0:
                cv2_imshow(res_plotted)
                break

    c = cv2.waitKey(1)
    if c == 27:
        break

