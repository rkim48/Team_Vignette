import sys, os
from pocketsphinx import *
import pyaudio
import pygame, sys
from pygame.locals import *
import pygame.camera

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
while True:
    buf = stream.read(1024,exception_on_overflow = False)
    decoder.process_raw(buf, False, False)
    if decoder.hyp() != None:
        print("Keyword Detected:",decoder.hyp().hypstr)
        if decoder.hyp().hypstr == 'take a picture ':
            pygame.init()
            pygame.camera.init()
            cam = pygame.camera.Camera("/dev/video0",(1920,1080))
            cam.start()
            image= cam.get_image()
            pygame.image.save(image,'/home/pi/Desktop/test.jpg')
            cam.stop()   
        decoder.end_utt()
        decoder.start_utt()
