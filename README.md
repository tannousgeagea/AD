# Radar-based anomaly Detection of pedestrian behaviors in radar raw data #

In this project, we applied anomaly detection to pedestrian motion patterns in recording data in automotive radar. We distinguish between two types of data: normal data, which are all possible pedestrian movements that can be easily encountered in real road traffic (e.g., walking, jogging, talking on the phone, etc.); abnormal data, which are all possible pedestrian movements that are rarely seen in a real road traffic (e.g., falling, dancing, tripping, hitting by a car, etc.). 

We implemented advanced anomaly detection methods in Python using ML libraries PyTorch and TensorFlow, as well as simulating virtual environments aided by Ansys AVxcelerate-tools.

To use the models please either run the command on python:

'''
python main.py --modelName vae --source path/to/datasets batch_size 16 latent_dim 16 lr 0.0001 epochs 100
'''

or check the example of exp.ipynb

