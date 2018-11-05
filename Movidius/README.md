Directory including sample code to compile a custom Tensorflow model into graph files the stick can use. This particular model is trained on the MNIST dataset and can recognize images of digits using the NCS. 

The following link describes the process to compile Tensorflow networks: https://movidius.github.io/ncsdk/tf_compile_guidance.html

The train_mnist_model.py script will save the MNIST trained model. 

The inference_mnist_model.py script removes specific lines from the trained model such as import data code and dropout layers. When this script is run, the mnist_inference.graph file will be created to be used for compiling. 

I would run the train_mnist_model.py and inference_mnist_model.py in Google Colab since it gives you a free GPU. 

The run.py script will perform inference on the data in the MNIST data using the NCS. 

