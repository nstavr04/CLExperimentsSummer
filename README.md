# Continual Learning on Edge using TensorFlow-Lite - Offline Experiments

This repository contains the code for the offline experiments performed in Continual Learning on Edge using TensorFlow-Lite.

Old and new offline experiments are used for reference. We utilize the experiments contained in newestOfflineExperiments folder.

The experiments are controlled from the controller.py class.

Experiments can be run using the following command and replacing the buffer size with the desired size:
(Make sure to comment / uncomment the desired experiments. Also do not run multiple experiments at the same time since memory issues may occur)

python controller.py --exp_RBS_3000

The experiments.py defines the functions used to evaluate (train) the model
The models.py defines the model structure as well as the functions for storing the feature patterns in the replay buffer.

utils.py and data_loader.py are helper classes used for the CORe50 dataset.

avalanche-core50-benchmark.py and cir_utils.py are taken and slightly modified from the CIR paper in an attempt to create a custom CORe050 dataset. More changes are needed.

CustomLearningRateScheduler.py is used to define the learning rate scheduler used in the experiments for Explonential LR Decay (ELRD).

In the experiments folder there are the graphs and the .json files from different experiments that have been run.

## More Useful Information:

- Tensorflow: Make sure to use Tensorflow 2.10.* since versions later than 2.10 do not offer GPU support on native-Windows.

- For enabling the GPU capabilities, it is best to use miniconda to create a venv. 
Instructions for the miniconda setup and additional files needed for GPU support can be seen on the following video:
https://www.youtube.com/watch?v=hHWkvEcDBO0

- The experiments were ran on the UCY HPC cluster. See the HPC file for help on the commands etc.

- CORe50 dataset: The dataset can be downloaded from the following link: https://vlomonaco.github.io/core50/index.html#download
We are utilizing the 128x128 images and the CORe50 NICv2 - 391 scenario. Make sure to add the dataset as a subdirectory inside CORe50-Dataset directory.

- Specifics to each function can be found in the code.

- The different algorithms etc. can be found in the code, some are commented out. You can swap between them by commenting / uncommenting the desired ones.

- In the on-device-training folder in newOfflineExperiments are the different versions of the script used to convert the 
tensorflow model to the .tflite file which we use to add in the android studio application.

- Use the generate_training_model.py / generate_training_model_2.py to generate the training model. As the current state of the
code offered by tensorflow for the generation of the model, you cannot change the names of the functions offered, it seems to cause issues
that lead to not being able to convert the model.