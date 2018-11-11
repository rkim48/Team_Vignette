The Makefile comes from https://github.com/ashwinvijayakumar/ncappzoo/blob/flowers/apps/flowers/Makefile and is used to retrain MobileNetv1 on the flowers dataset from Tensorflow tutorials.

This tutorial will show you how to retrain the MobileNetv1 model on an apples and oranges dataset and compile the model into a graph file usable by the Movidius NCS. To retrain the model on your own custom dataset, scroll further down. 

Make sure you are on a Linux environment! 

Step 1: Download ncappzoo(git clone -b ncsdk2 https://github.com/movidius/ncappzoo.git)  
Step 2: cd to ncappzoo/tensorflow  
Step 3: Copy/paste pg folder inside ncappzoo/tensorflow directory  
Step 4: Copy/paste download_and_convert_data.py file into tensorflow/tf_models/models/research/slim directory  
Step 5: Copy paste datasets folder into the same directory as above 

You are now setup to fine tune the MobileNetv1 model and compile the model into a Movidius graph file!

Step 0: Optional: To compile the model, download ncsdk(git clone -b ncsdk2 https://github.com/movidius/ncsdk.git)      
Step 1: cd to ncappzoo/tensorflow/pg  
Step 2: Run command: make all (this will convert images into tf_records format and produce label.txt for training)  
Step 3: Run command: make train (training the last layers of the model)  
Step 4: Run command: make all   
Step 5: Run command: make run (perform inference on test image in the inference_test folder)  

To use custom dataset, cd to ncappzoo/tensorflow/data_photos and place folders with each class of images inside. Additionally, you can place a photo not in the training data inside the inference_test folder if you want to test the new model. 
The next part will be to edit some Python dependencies to accomadate for the new training data.
Coming Soon!
