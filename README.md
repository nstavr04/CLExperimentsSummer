# Continual Learning on Edge using TensorFlow-Lite - Offline Experiments

This repository contains the code for the offline experiments performed in Continual Learning on Edge using TensorFlow-Lite.

Old and new offline experiments are used for reference. We utilize the experiments contained in newestOfflineExperiments folder.

The experiments are controlled from the controller.py class.
The experiments.py defines the functions used to evaluate (train) the model
The models.py defines the model structure as well as the functions for storing the feature patterns in the replay buffer.

utils.py and data_loader.py are helper classes used for the CORe50 dataset.

## Useful Information:

- Tensorflow: Make sure to use Tensorflow 2.10.* since versions later than 2.10 do not offer GPU support on native-Windows.

- For enabling the GPU capabilities, it is best to use miniconda to create a venv. 
Instructions for the miniconda setup and additional files needed for GPU support can be seen on the following video:
https://www.youtube.com/watch?v=hHWkvEcDBO0

- CORe50 dataset: The dataset can be downloaded from the following link: https://vlomonaco.github.io/core50/index.html#download
We are utilizing the 128x128 images and the CORe50 NICv2 - 391 scenario.

- Specifics to each function can be found in the code.

- In the on-device-training folder in newOfflineExperiments are the different versions of the script used to convert the 
tensorflow model to the .tflite file which we use to add in the android studio application.

- Use the generate_training_model.py / generate_training_model_2.py to generate the training model. As the current state of the
code offered by tensorflow for the generation of the model, you cannot change the names of the functions offered, it seems to cause issues
that lead to not being able to convert the model.