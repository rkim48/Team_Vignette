import sys, os
import datetime
from pocketsphinx import *
import pyaudio
import pygame
from pygame.locals import *
import pygame.camera
import label_image as lab
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import time
import argparse

modeldir = "/home/pi/PocketSphinx_Dev/pocketsphinx-5prealpha/model"

# Create a decoder with certain model
config = Decoder.default_config()
config.set_string('-hmm', os.path.join(modeldir, 'en-us/en-us'))
config.set_string('-dict', os.path.join(modeldir, 'en-us/cmudict-en-us.dict'))
config.set_string('-kws', '/home/pi/Vignette/Command Thresholds/Command_Threshold_layer_1.txt')


#Creating second voice layer
config_2 = Decoder.default_config()
config_2.set_string('-hmm', os.path.join(modeldir, 'en-us/en-us'))
config_2.set_string('-dict', os.path.join(modeldir, 'en-us/cmudict-en-us.dict'))
config_2.set_string('-kws', '/home/pi/Vignette/Command Thresholds/Command_Threshold_layer_2.txt')



p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
stream.start_stream()

# Process audio chunk by chunk. On keyword detected perform action and restart search
decoder = Decoder(config)
decoder.start_utt()

#Creating Tensorflow Model
IM_WIDTH = 300    
IM_HEIGHT = 300 

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

# This is needed since the working directory is the object_detection folder.
os.chdir("/home/pi/tensorflow1/models/research/object_detection")
sys.path = ['', '/usr/lib/python35.zip', '/usr/lib/python3.5', '/usr/lib/python3.5/plat-arm-linux-gnueabihf', '/usr/lib/python3.5/lib-dynload', '/home/pi/.local/lib/python3.5/site-packages', '/usr/local/lib/python3.5/dist-packages', '/usr/local/lib/python3.5/dist-packages/protobuf-3.5.1-py3.5-linux-armv7l.egg', '/usr/lib/python3/dist-packages']
sys.path.append("/home/pi/tensorflow1/models/research/object_detection")
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'product_model'

# Grab path to current working directory
CWD_PATH = '/home/pi/tensorflow1/models/research/object_detection'

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','product_labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 4

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

print("Detection Started")
while True:
    buf = stream.read(1024,exception_on_overflow = False)
    decoder.process_raw(buf, False, False)
    if decoder.hyp() != None:
        
        print("Raspberry Detected")
        decoder.end_utt()
        decoder_2 = Decoder(config_2)
        decoder_2.start_utt()
        
        while True:
            
            buf_2 = stream.read(1024,exception_on_overflow = False)
            decoder_2.process_raw(buf_2, False, False)
        
            if decoder_2.hyp() != None:
            
                print("Keyword Detected:",decoder_2.hyp().hypstr)
                if decoder_2.hyp().hypstr == 'classify ' or decoder_2.hyp().hypstr == 'take a picture ':
                    start = time.time()
                    time.clock()
                    elapsed = 0
                    camera = cv2.VideoCapture(0)
                    ret = camera.set(3,IM_WIDTH)
                    ret = camera.set(4,IM_HEIGHT)
                    while(elapsed < 240):
                        
                        t1 = cv2.getTickCount()

                        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
                        # i.e. a single-column array, where each item in the column has the pixel RGB value
                        ret, frame = camera.read()
                        frame_expanded = np.expand_dims(frame, axis=0)

                        # Perform the actual detection by running the model with the image as input
                        (boxes, scores, classes, num) = sess.run(
                            [detection_boxes, detection_scores, detection_classes, num_detections],
                            feed_dict={image_tensor: frame_expanded})
                    
                        
                        top_scores = np.where(scores > 0.85)
                        classes_recognized = classes[top_scores]
                        print(classes_recognized)
                        labels = ['crest','dawn','head shoulders','tide']
                        
                        if len(classes_recognized) > 0:
                            csv_data = pd.read_csv('/home/pi/Vignette Data/item_frequency.csv', header= None).values
                            for item in classes_recognized:
                                class_label = labels[int(item)-1]
                                itemindex = np.where(csv_data[:,0]==class_label)
                                csv_data[itemindex,1] = csv_data[itemindex,1] + 1
                            csv_data = pd.DataFrame(csv_data)
                            csv_data.to_csv('/home/pi/Vignette Data/item_frequency.csv',index=False, header = None)
                        
                        
                        
                        
                        # Draw the results of the detection (aka 'visulaize the results')
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            frame,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            category_index,
                            use_normalized_coordinates=True,
                            line_thickness=8,
                            min_score_thresh=0.85)

                        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
                        
                        # All the results have been drawn on the frame, so it's time to display it.
                        cv2.imshow('Object detector', frame)

                        t2 = cv2.getTickCount()
                        time1 = (t2-t1)/freq
                        frame_rate_calc = 1/time1
                        
                        elapsed = time.time() - start
                        # Press 'q' to quit
                        if cv2.waitKey(1) == ord('q'):
                            break

                    camera.release()

                    cv2.destroyAllWindows()
                
                #Restarts voice decoding  
                decoder_2.end_utt()
                #decoder.end_utt()
                decoder.start_utt()
                break