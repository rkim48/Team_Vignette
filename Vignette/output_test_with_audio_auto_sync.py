# Testing

import os
import cv2
import sys
import numpy as np
import ntpath
import argparse
import skimage.io
import skimage.transform
import mvnc.mvncapi as mvnc
from draw_box import *
from datetime import *
import pytz
import pyaudio
import pygame
from pygame.locals import *
from pocketsphinx import *
import pygame.camera
import data_sync_azure as dsa
import insert_image as ins_img
import pandas as pd
import tensorflow as tf
import time
import argparse
import speech_recognition as sr
import subprocess
from subprocess import call




#from utils import visualize_output
#from utils import deserialize_output

# Detection threshold: Minimum confidance to tag as valid detection
CONFIDANCE_THRESHOLD = 0.70 # 60% confidant

# Variable to store commandline arguments
ARGS                 = None
cam = None

# ---- Step 1: Open the enumerated device and get a handle to it -------------

def open_ncs_device():

    devices = mvnc.enumerate_devices()
    if len( devices ) == 0:
        print( "No devices found" )
        quit()

    # Get a handle to the first enumerated device and open it
    device = mvnc.Device( devices[0] )
    device.open()

    return device

# ---- Step 2: Load a graph file onto the NCS device -------------------------

def load_graph( device ):

    # Read the graph file into a buffer
    with open( ARGS.graph, mode='rb' ) as f:
        blob = f.read()

    # Load the graph buffer into the NCS
    graph = mvnc.Graph( ARGS.graph )
    # Set up fifos
    fifo_in, fifo_out = graph.allocate_with_fifos(device, blob)

    return graph, fifo_in, fifo_out

# ---- Step 3: Pre-process the images ----------------------------------------

old_time = time.time()
use_time = time.time()
item_dict = {"bounce":0, "febreeze":0, "gain":0, "tide":0}
old_coord = [0,0]
startup = True
class_pause_flag = False


def infer_image(graph, fifo_in, fifo_out ):

    # Grab a frame from the camera
    ret, frame = cam.read()
    height, width, channels = frame.shape

    # Resize image [Image size if defined by choosen network, during training]
    input_image = cv2.resize(frame, tuple(ARGS.dim), cv2.INTER_LINEAR)
        
    #input_image = input_image.astype(np.float32)
    input_image = input_image[:, :, ::-1]
    input_image = np.divide(input_image, 255.0)
      # convert to RGB


    # Load tensor and get result.  This executes the inference on the NCS
    graph.queue_inference_with_fifo_elem(fifo_in, fifo_out, input_image.astype(np.float32), 'user object')
    output, userobj = fifo_out.read_elem()
    
    input_height = 416
    input_width = 416
    score_threshold = 0.07
    iou_threshold = 0.07
    
    output_image, pg_class, coords = postprocessing(output, input_image, score_threshold, iou_threshold, 416, 416)
    
    global old_time, old_coord, startup, use_time, class_pause_flag, item_dict
    item = None
    
    

    coord_thresh = 100
    if class_pause_flag == True & abs((time.time() - use_time) > 5):
        class_pause_flag = False
        print("Ready to detect next use!")
    if ((time.time() - old_time) > 1.5) & (len(pg_class) > 0) & (len(coords)>0):
        new_coord = coords
        #print(coords)
        #print(abs(new_coord-old_coord),coord_thresh)
        distance = ((new_coord[0]-old_coord[0])**2 + (new_coord[1]-old_coord[1])**2)**(0.5)
        #print(distance)
        if (distance > coord_thresh) & (startup == False) & (class_pause_flag==False):
            print("Use of {} detected! Coordinate movement of:{}".format(pg_class[0],distance))
            if pg_class[0] == 'bounce':
                #item_dict['bounce'] += 1
                item = 'bounce'
            elif pg_class[0] == 'febreeze':
                #item_dict['febreeze'] += 1
                item = 'febreeze'
            elif pg_class[0] == 'gain':
                #item_dict['gain'] += 1
                item = 'gain'
            elif pg_class[0] == 'tide':
                #item_dict['tide'] += 1
                item = 'tide'
            else:
                item = None
                
            #print(item_dict)
                
            
        
            
            class_pause_flag = True
            use_time = time.time()
        old_time = time.time()
        startup = False
        old_coord = new_coord
            
    
    
    return input_image, frame, output_image, item

# ---- Step 4: Read & print inference results from the NCS -------------------
    
    
    
#def infer_image( graph, img, frame, fifo_in, fifo_out ):
#
#    # Load the labels file 
#    labels =[ line.rstrip('\n') for line in 
#                   open( ARGS.labels ) if line != 'classes\n'] 
#
#    # Load the image as a half-precision floating point array
#    graph.queue_inference_with_fifo_elem( fifo_in, fifo_out, img.astype(numpy.float32), None )
#
#    # Get the results from NCS
#    output, userobj = fifo_out.read_elem()
    
    
def clean_up(device, graph, fifo_in, fifo_out):
    fifo_in.destroy()
    fifo_out.destroy()
    graph.destroy()
    device.close()
    device.destroy()
    cam.release()
    cv2.destroyAllWindows()

def capture_img(pic_name):
    #Takes photo using fswebcam command
    call(["fswebcam", "-d","/dev/video0", "-r", "300x300", "--no-banner", pic_name])

def main():
    
    # Create a pocketsphinx decoder with certain model
    modeldir = "/home/pi/PocketSphinx_Dev/pocketsphinx-5prealpha/model"
    config = Decoder.default_config()
    config.set_string('-hmm', os.path.join(modeldir, 'en-us/en-us'))
    config.set_string('-dict', os.path.join(modeldir, 'en-us/cmudict-en-us.dict'))
    config.set_string('-kws', '/home/pi/Vignette/Command Thresholds/Command_Threshold_layer_1.txt')

    #Setting up Sphinx Audio Stream
    p = pyaudio.PyAudio()
    stream_phinx = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    stream_phinx.start_stream()

    # Process audio chunk by chunk. On keyword detected perform action and restart search
    decoder = Decoder(config)
    decoder.start_utt()
    
    #Setting up Movidius and graph
    device = open_ncs_device()
    graph, fifo_in, fifo_out = load_graph( device )
    
    print("Detection Started")
    
    while( True ):
        
        buf = stream_phinx.read(1024,exception_on_overflow = False)
        decoder.process_raw(buf, False, False)
        if decoder.hyp() != None:
            print("Raspberry Detected")
            
            #Ending Sphinx Speech
            decoder.end_utt()
            stream_phinx.stop_stream()
            stream_phinx.close()
            p.terminate()
            
            #Setting up Speech Model from Azure
            r = sr.Recognizer()
            mic = sr.Microphone(device_index=0)
            
            with mic as source:
                r.adjust_for_ambient_noise(source) 
                print("Say something!")
                audio = r.listen(source,phrase_time_limit = 3)

            BING_KEY = '1288b02ddf6c46ce81935ce0c2ec22bd'
            
            try:
                speech_to_text_result = r.recognize_bing(audio, key=BING_KEY)
            except:
                speech_to_text_result = "Nothing"
            
            print(speech_to_text_result)
            try:
                print("Microsoft Bing Voice Recognition thinks you said: " + speech_to_text_result)
            except sr.UnknownValueError:
                print("Microsoft Bing Voice Recognition could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from Microsoft Bing Voice Recognition service; {0}".format(e))
            except:
                None

            if 'Laundry' in speech_to_text_result or 'laundry' in speech_to_text_result:
                
                print("Starting Laundry Detection!")
                start_elap = time.time()
                time.clock()
                #elapsed = 0
                global cam
                cam = cv2.VideoCapture( ARGS.video )
                
                while(True):
                    
                    n_frames = 0
                    seconds = 0.0
                    fps = 0.0
                    start = time.time()
                    n_frames = n_frames + 1
                    
                    img, frame, output_image, rec_item = infer_image(graph, fifo_in, fifo_out)
                    
                    end = time.time()
                    seconds = (end - start)
                    fps  = ( fps + (1/seconds) ) / 2
                    
                    output_image = output_image[:, :, ::-1]
                    output_image = np.ascontiguousarray(output_image, dtype=np.float32)
                    output_image = cv2.resize(output_image,(1200,800))
                    #cv2.putText(output_image,str(format(fps,'.2f')),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                    cv2.imshow('Video', output_image)
                    
                    
                    # Display the frame for 5ms, and close the window so that the next frame 
                    # can be displayed. Close the window if 'q' or 'Q' is pressed.
                    labels = ['tide','gain','bounce','febreeze']
                    
                    #print(rec_item)
                    
                    if  rec_item != None:
                        csv_data = pd.read_csv('/home/pi/Vignette Data/item_frequency.csv')
                        
                        #class_label = labels[int(item)-1]
                        index = csv_data.index[csv_data['Key'] == rec_item]
                        print(index)
                        csv_data.loc[index,'Value'] = csv_data.loc[index,'Value'] + 1
                        #itemindex = np.where(csv_data[:,0]==class_label)
                        #csv_data[itemindex,1] = csv_data[itemindex,1] + 1
                        csv_data = pd.DataFrame(csv_data)
                        csv_data.to_csv('/home/pi/Vignette Data/item_frequency.csv',index=False)
                        
                        dsa.main()
                    
                    
##                    if( cv2.waitKey( 1 ) & 0xFF == ord( 'q' ) ):
##                        #os.remove('yolo_inferences.csv')
##                        with open( 'yolo_inferences.csv', 'a', newline='' ) as csvfile:
##
##                            inference_log = csv.writer( csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL )
##                            inference_log.writerow(['Key','Value'])
##                            for key in item_dict.keys():
##                                inference_log.writerow([key,item_dict[key]])
##                        break
                                
                    
                    #elapsed = time.time() - start_elap
                    
                    # Display the frame for 5ms, and close the window so that the next frame 
                    # can be displayed. Close the window if 'q' or 'Q' is pressed.
                    if( cv2.waitKey( 1 ) & 0xFF == ord( 'q' ) ):
                        break
                
                cam.release()
                cv2.destroyAllWindows()    
                #clean_up(device, graph, fifo_in, fifo_out)    
                
            if 'send my data' in speech_to_text_result or 'Send my data' in speech_to_text_result:
                dsa.main()
                ins_img.main()
                print("Data sent")
            
            if 'delete my data' in speech_to_text_result or 'Delete my data' in speech_to_text_result:
                
                #Deleting CSV data
                csv_data = pd.read_csv('/home/pi/Vignette Data/item_frequency.csv')
                csv_data.iloc[0:,1] = 0
                csv_data.to_csv('/home/pi/Vignette Data/item_frequency.csv',index=False)
                
                #Deleting photos
                for file in os.listdir('/home/pi/yolo-darkflow-movidius/images'):
                    if file.endswith('.jpg'):
                        os.remove('/home/pi/yolo-darkflow-movidius/images/' + file)
                        
                dsa.main()
                ins_img.main()

                print("Data Deleted")
                    
            if 'take a picture' in speech_to_text_result or 'Take a picture' in speech_to_text_result:
                
                utc=pytz.UTC
                now = datetime.now()
                now_str = now.strftime("%Y-%m-%d_%H:%M:%S")
                pic_name = '/home/pi/yolo-darkflow-movidius/images/' + now_str +'.jpg'
                print(pic_name)
                capture_img(pic_name)
                print("Picture Saved")
                
##                cap = cv2.VideoCapture( ARGS.video )
##                return_value, image = cap.read()
##                now = datetime.datetime.now()
##                pic_name = '/home/pi/yolo-darkflow-movidius/images/' + str(now) +'.jpg'
##                cv2.imwrite(pic_name,image)
##                del(cap)
              
            #Restarting Audio
            p = pyaudio.PyAudio()
            stream_phinx = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
            stream_phinx.start_stream()
            decoder.start_utt()
            
    clean_up(device, graph, fifo_in, fifo_out)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
                         description="Image classifier using Intel® Movidius™ Neural Compute Stick." )

    parser.add_argument( '-g', '--graph', type=str,
                         default='built_graph/tiny-yolo-voc-4c.graph',
                         help="Absolute path to the neural network graph file." )

    parser.add_argument( '-l', '--labels', type=str,
                         default='labels.txt',
                         help="Absolute path to labels file." )

    parser.add_argument( '-M', '--mean', type=float,
                         nargs='+',
                         default=[78.42633776, 87.76891437, 114.89584775],
                         help="',' delimited floating point values for image mean." )

    parser.add_argument( '-S', '--scale', type=float,
                         default=1,
                         help="Absolute path to labels file." )

    parser.add_argument( '-D', '--dim', type=int,
                         nargs='+',
                         default=[416, 416],
                         help="Image dimensions. ex. -D 224 224" )

    parser.add_argument( '-c', '--colormode', type=str,
                         default="RGB",
                         help="RGB vs BGR color sequence. \
                               Defined during model training." )

    parser.add_argument( '-v', '--video', type=int,
                         default=0,
                         help="Index of your computer's V4L2 video device. \
                               ex. 0 for /dev/video0" )
    
    ARGS = parser.parse_args()
    
    #cam = cv2.VideoCapture( ARGS.video )
    
    main()