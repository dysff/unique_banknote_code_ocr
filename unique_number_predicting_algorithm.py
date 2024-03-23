import cv2
import numpy as np
import tensorflow as tf
import easyocr

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
session = tf.compat.v1.InteractiveSession(config=config)

DATA_PATH = 'rcnn_evaluation_data/0.jpg'
model = tf.keras.models.load_model('banknote_rcnn_4.h5')

#--------------------------BBOX PREDICTING--------------------------

def bounding_box_transformer(y_pred):
  x1 = y_pred[2] / 2 + y_pred[0]
  y1 = y_pred[1]
  width1 = y_pred[2] / 2
  height1 = y_pred[3] / 2

  x2 = width1 / 3.5 + x1
  y2 = height1 / 4 + y1
  width2 = width1
  height2 = height1 / 2

  y_pred = [x2, y2, width2, height2]
  
  return y_pred

img = cv2.imread(DATA_PATH)
img_resized = np.array(cv2.resize(img, [170, 128]))
image_to_predict = np.expand_dims(img_resized, axis=0)

bbox_predicted = bounding_box_transformer(model.predict(image_to_predict).tolist()[0])

#From relative coords to absolute
(h, w) = img_resized.shape[:2]
adjusted_bbox = [bbox_predicted[0] * w,#top_x
                 bbox_predicted[1] * h,#top_y
                 bbox_predicted[2] * w,#top_x
                 bbox_predicted[3] * h]#top_y

#Bbox transformation with original image ratio
width_ratio = img.shape[1] / 170
height_ratio = img.shape[0] / 128

#Scaling
bbox_predicted = [adjusted_bbox[0] * width_ratio,
                  adjusted_bbox[1] * height_ratio,
                  adjusted_bbox[2] * width_ratio,
                  adjusted_bbox[3] * height_ratio]

#Crop the region of interest
def crop_image(img, bbox_data):
  x = int(bbox_data[0])
  y = int(bbox_data[1])
  width = int(bbox_data[2])
  height = int(bbox_data[3])
  
  cropped_img = img[y:y+height, x:x+width]
  
  return cropped_img

roi = crop_image(img, bbox_predicted)

#Visualisation
import matplotlib.pyplot as plt

plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
plt.show()

#--------------------------EASYOCR--------------------------

def text_recognize(img):
    reader = easyocr.Reader(["ru"], gpu=True)
    result = reader.readtext(img, detail=1)

    return result

predicted_text = text_recognize(roi)
predicted_filtered_text = ''
print(predicted_text)

#Set confidence threshold
for (bbox, text, prob) in predicted_text:
  
  if prob >= 0.75:
    text = text.replace(' ', '')
    predicted_filtered_text += text
    
print(predicted_filtered_text)