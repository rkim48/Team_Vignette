import sys, os
import subprocess as sp
from pocketsphinx import *
import pyaudio
import pygame, sys
from pygame.locals import *
import pygame.camera
import label_image as lab
import numpy as np
import tensorflow as tf

modeldir = "/home/pi/PocketSphinx_Dev/pocketsphinx-5prealpha/model"

# Create a decoder with certain model
config = Decoder.default_config()
config.set_string('-hmm', os.path.join(modeldir, 'en-us/en-us'))
config.set_string('-dict', os.path.join(modeldir, 'en-us/cmudict-en-us.dict'))
config.set_string('-kws', '/home/pi/PocketSphinx_Dev/pocketsphinx-5prealpha/Command_Threshold.list')

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
stream.start_stream()

# Process audio chunk by chunk. On keyword detected perform action and restart search
decoder = Decoder(config)
decoder.start_utt()

#Creating Tensorflow Model
graph = lab.load_graph('/home/pi/Senior Design/Other Files/output_graph.pb')
labels = lab.load_labels('/home/pi/Senior Design/Other Files/output_labels.txt')
print("Detection Started")
while True:
    buf = stream.read(1024,exception_on_overflow = False)
    decoder.process_raw(buf, False, False)
    if decoder.hyp() != None:
        print("Keyword Detected:",decoder.hyp().hypstr)
        if decoder.hyp().hypstr == 'take a picture ':
            
            #Taking Photo with Camera
            pygame.init()
            pygame.camera.init()
            cam = pygame.camera.Camera("/dev/video0",(224,224))
            cam.start()
            image= cam.get_image()
            pygame.image.save(image,'/home/pi/Desktop/class.jpg')
            cam.stop()
            
            #Classifiying Image
            tense = lab.read_tensor_from_image_file('/home/pi/Desktop/class.jpg',input_height = 224, input_width = 224)
            
            input_name = "import/" + 'Placeholder'
            output_name = "import/" + 'final_result'
            input_operation = graph.get_operation_by_name(input_name)
            output_operation = graph.get_operation_by_name(output_name)
            
            with tf.Session(graph=graph) as sess:
                results = sess.run(output_operation.outputs[0], {
                    input_operation.outputs[0]: tense
                })
            results = np.squeeze(results)

            top_k = results.argsort()[-5:][::-1]

            for i in top_k:
                print(labels[i], results[i])
            
        #Restarts voice decoding   
        decoder.end_utt()
        decoder.start_utt()