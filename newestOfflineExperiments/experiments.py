from models import ContinualLearningModel
from data_loader import CORE50
from CustomLearningRateScheduler import CustomLearningRateScheduler
from utils import *
import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
from matplotlib import pyplot as plt
import json
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# DATASET_ROOT = 'C:/Users/nikol/Desktop/University/Year-4/ADE/ThesisCodeExperiments/CORe50-Dataset/core50_128x128/'
# DATASET_ROOT = 'D:/MyFiles/ADE/SummerCodeContinuation/CORe50-Dataset2/core50_128x128/'
DATASET_ROOT = '/home/nstavr04/clofflineexperiments/CLExperimentsSummer/CORe50-Dataset/core50_128x128'

class Experiments:

    def __init__(self):
        print("> Experiments Initialized")

    def plotExperiment(self,experiment_name,title):
        min_val = 100
        max_val = 50
        with open('experiments/' + experiment_name + '.json', ) as json_file:
            usecases = json.load(json_file)
            for usecase in usecases:
                for key, value in usecase.items():
                    plt.plot(value['acc'], label=key)
                    cur_min = min(value['acc'])
                    cur_max = max(value['acc'])
                    if cur_min < min_val:
                        min_val = cur_min
                    if cur_max > max_val:
                        max_val = cur_max

        plt.title(title)
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Encountered Batches")
        plt.yticks(np.arange(round(min_val), round(max_val)+10, 5))
        plt.grid()
        plt.legend(loc='best')
        #plt.show()
        plt.savefig(experiment_name)
    
    # Function used to store our experiment results in a json file
    def storeExperimentOutputNew(self, experiment_name, usecase_name, accuracies, losses):
        data = []

        # Load previously recorded usescases
        with open('experiments/' + experiment_name + '.json', ) as json_file:
            data = json.load(json_file)

        # Store new usecase
        exp = dict()
        exp[usecase_name] = dict()
        exp[usecase_name]["acc"] = accuracies
        exp[usecase_name]["loss"] = losses
        data.append(exp)

        # Write the updated data back to the file
        with open('experiments/' + experiment_name + '.json', 'w') as outfile:
            json.dump(data, outfile)

    def print_trainable_status(self, model):
        for layer in model.layers:
            print(f"{layer.name}: {layer.trainable}")

    # New implementation with hidden layers
    def runHiddenLayersExperiment(self, experiment_name, usecase, replay_size, num_hidden_layers):
        print("> Running Hidden Layers experiment")

        dataset = CORE50(root=DATASET_ROOT, scenario="nicv2_391", preload=False)
        test_x, test_y = dataset.get_test_set()
        test_x = preprocess(test_x)

        # Just a print of the test set to know the distribution of classes
        counter_dict = Counter(test_y)
        sorted_counter_dict = {k: counter_dict[k] for k in sorted(counter_dict)}
        print("Test_Set_Class_Distribution: ", sorted_counter_dict)

        # Building main model
        cl_model = ContinualLearningModel(image_size=128, name=usecase,replay_buffer=replay_size)
        cl_model.buildBaseHidden(hidden_layers=num_hidden_layers)
        cl_model.buildHeadHidden(sl_units=128, hidden_layers=num_hidden_layers)
        cl_model.buildCompleteModel()

        #### Used for debugging ####

        # # After building the complete model
        print("Base model trainable status:")
        self.print_trainable_status(cl_model.base)

        print("\nHead model trainable status:")
        self.print_trainable_status(cl_model.head)

        print("\nComplete model trainable status:")
        self.print_trainable_status(cl_model.model)

        print("\nComplete model summary:")
        cl_model.model.summary()

        # # Stop the program
        # exit()

        #### End of debugging ####

        # Used for exponential learning decay
        lr_schedule = CustomLearningRateScheduler(initial_learning_rate=0.001, gamma=0.9999846859337639)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

        accuracies = []
        losses = []

        # Training, loop over the training incremental batches
        for i, train_batch in enumerate(dataset):
            train_x, train_y = train_batch
            train_x = preprocess(train_x)

            print("----------- batch {0} -------------".format(i))
            print("train_x shape: {}, train_y shape: {}"
                  .format(train_x.shape, train_y.shape))

            current_lr = lr_schedule(i).numpy()
            print("Current learning rate: ", current_lr)

            if i == 1:
                # Previous values on both: 0.00005
                # A higher learning rate on the head such as 0.001 works way better especially with the latent replay buffer
                # Increasing the learning rate even more to e.g. 0.01 makes the model unstable since on some batches we have huge loss
                # but the overall accuracy remains more or less the same
                cl_model.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.00005),
                                       loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                cl_model.head.compile(optimizer=optimizer,
                                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Padding of the first batch. Unsure about this
            if i == 0:
                (train_x, train_y), it_x_ep = pad_data([train_x, train_y], 128)
            
            shuffle_in_unison([train_x, train_y], in_place=True)

            print("---------------------------------")

            # This would be how to augment data and last parameter is how many to produce
            # The augmentation is random meaning we don't know which train_x samples will be used everytime to generate new samples
            # We should look into how to achieve class balancing in the augmentation
            # Also, after augmenting we need to do feature extraction.
            # This means in the replay buffer we cant augment if we save the features of the samples
            # The augmentation needs to happen on the original images
            # augmented_train_x, augmented_train_y = augment_data(train_x, train_y, 100)

            features = cl_model.feature_extractor.predict(train_x)

            # Combining the new samples and the replay buffer samples before training
            print("> Combining new samples and replay buffer samples before training")
            if i >= 1:
                # Get replay samples
                replay_x = np.array(cl_model.replay_representations_x)
                replay_y = np.array(cl_model.replay_representations_y)

                # Combine new samples with replay samples
                combined_x = np.concatenate((features, replay_x), axis=0)
                combined_y = np.concatenate((train_y, replay_y), axis=0)
            else:
                combined_x = features
                combined_y = train_y

            # Shuffle the combined samples
            shuffle_in_unison([combined_x, combined_y], in_place=True)

            print("combined-x shape: {}, combined-y shape: {}".format(combined_x.shape, combined_y.shape))
            
            # Debug stuff
            print("Expected input shape for head: ", cl_model.head.layers[0].input_shape)

            # Fit the head on the combined samples
            cl_model.head.fit(combined_x, combined_y, epochs=4, verbose=0)

            #### Used if we want to fit the head on the new samples and the replay buffer samples separately ####

            # cl_model.head.fit(features, train_y, epochs=4, verbose=0)
            # if i >= 1:
            #     cl_model.replay()
            # cl_model.storeRepresentations(train_x, train_y)

            #### End of the above ####

            # Store the representations of the new samples in the replay buffer
            # cl_model.storeRepresentationsNativeRehearsal(train_x, train_y, i+1)

            cl_model.BRS(features, train_x, train_y, i+1)

            # Kind of tricky to obtain each training loss, we can't with fit so we need to loop each sample. Will leave it for now
            # cl_model.LARS(train_x, train_y, i+1, passthelosshere)

            # Evaluate the model on the test set
            loss, acc = cl_model.model.evaluate(test_x, test_y)
            accuracies.append(round(acc*100,1))
            losses.append(loss)
            print("> ", cl_model.name, " Accuracy: ", acc, " Loss: ", loss)
            print("---------------------------------")

        # Store results in json file
        self.storeExperimentOutputNew(experiment_name=experiment_name,
                                   usecase_name=usecase,
                                   accuracies=accuracies,
                                   losses=losses)

    # Function to augment data used for both the training and replay samples    
    def augment_data(data_x, data_y, batch_size):
        # Initialize the data generator
        # Need to experiment on the values
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest')
        
        # Generate augmented data
        data_gen = datagen.flow(data_x, data_y, batch_size=batch_size)
        
        # Get a batch of augmented data
        augmented_data_x, augmented_data_y = next(data_gen)
        
        return augmented_data_x, augmented_data_y