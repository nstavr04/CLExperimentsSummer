import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import numpy as np
import random
import gc
from collections import Counter

from CustomLearningRateScheduler import CustomLearningRateScheduler


class ContinualLearningModel:

    def __init__(self,image_size=224,name="None",replay_buffer=3000):
        print("> Continual Learning Model Initiated")
        # base = bases.MobileNetV2Base(image_size=224)
        self.image_size = image_size
        self.name = name
        self.replay_representations_x = []
        self.replay_representations_y = []
        # Only used for LARS
        self.replay_representations_losses = []
        self.replay_buffer = replay_buffer # The number of patterns stored

        # self.lr_schedule = CustomLearningRateScheduler(initial_learning_rate=0.004, gamma=0.9999846859337639)
        # self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

    # Base of our model. We remove the last N layers of MobileNetV2    
    def buildBaseHidden(self,hidden_layers=0):
        baseModel = tf.keras.applications.MobileNetV2(input_shape=(self.image_size, self.image_size, 3),
                                                      alpha=1.0,
                                                      include_top=False,
                                                      weights='imagenet')
        
        # Batch normalization layers replaced with bactch renormalization layers - Better for CL
        for l in baseModel.layers:
            if ('_BN' in l.name):
                l.renorm = True
        
        baseModel.trainable = False

        base_model_truncated = tf.keras.Model(inputs=baseModel.input, outputs=baseModel.layers[-hidden_layers-1].output)
        self.base = base_model_truncated

        inputs = tf.keras.Input(shape=(self.image_size, self.image_size, 3))
        f = inputs
        f_out = self.base(f)
        self.feature_extractor = tf.keras.Model(f, f_out)
        self.feature_extractor.compile(optimizer=self.optimizer, loss='categorical_crossentropy',
                                       metrics=['accuracy'])

    # Head of our model. We add N layers to the end of MobileNetV2 + a dense layer + softmax layer
    def buildHeadHidden(self, sl_units=32, hidden_layers=0):

        baseModel = tf.keras.applications.MobileNetV2(input_shape=(self.image_size, self.image_size, 3),
                                                      alpha=1.0,
                                                      include_top=False,
                                                      weights='imagenet')

        self.sl_units = sl_units

        # Create a new head model
        self.head = tf.keras.Sequential()

        # Add the last N layers of MobileNetV2 to the head model
        for i in range(-hidden_layers, 0):
            layer = baseModel.layers[i]
            layer.trainable = True
            self.head.add(layer)

        self.head.add(layers.Flatten(input_shape=(4, 4, 1280)))
        # Removed the dense layer since we add Hidden Layers from MobileNetV2 now
        self.head.add(layers.Dense(
        units=sl_units,
        activation='relu',
        kernel_regularizer=l2(0.01),
        bias_regularizer=l2(0.01)
        ))

        # Softmax layer (Last layer)
        self.head.add(layers.Dense(
            units=50,  # Number of classes
            activation='softmax',
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01)),
        )

        # It's worth noting that the compiling of the head and the model here is probably redundant
        # since we compile the head and model again on the experiments function with a different learning rate
        self.head.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])    

    def buildCompleteModel(self):
        inputs = tf.keras.Input(shape=(self.image_size, self.image_size, 3))
        x = inputs
        x = self.base(x)
        outputs = self.head(x)
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    # Refreshing replay memory with new samples and removing old ones if necessary
    # Fixed bug with patterns being a little less than they should be
    # This bug I think that it is caused because the replacement batch size is calculated as a percentage of the replay buffer
    # Which is greater than the actual number of patterns stored in the replay buffer
    #### Old function ####
    def storeRepresentations(self, train_x, train_y):
        # We need to add replay representations and replacement batch size not train_x
        replacement_batch_size = int(self.replay_buffer * 0.015) # 1.5% sample replacement
        replay_memory_size = len(self.replay_representations_x) + replacement_batch_size

        # It's good to know that the first batch has around 3000+ samples 
        # The rest are around 300ish
        # The last few batches don't introduce new classes
        # If we want to look in depth each training batch contents we can check batches filelists in CORe50
        # https://vlomonaco.github.io/core50/index.html#download

        # For testing purposes
        # print("Replay Memory Size: ",replay_memory_size)
        # print("replay representation x: ",len(self.replay_representations_x))
        # print("train x: ",len(train_x))

        # A check to see if the replacement batch size is greater than the number of samples in a batch
        # This could happen for e.g buffer sizes of 30000 since the batch size is around 300 samples and 1.5% of 30000 is 450
        if replacement_batch_size > len(train_x):
            replacement_batch_size = train_x

        # If the replay buffer will overfill we need to remove some old samples
        if replay_memory_size >= self.replay_buffer:

            x_sample, y_sample = zip(*random.choices(list(zip(train_x, train_y)), k=replacement_batch_size))
            x_sample = self.feature_extractor.predict(np.array(x_sample))

            # Removing old samples
            patterns_to_delete = random.sample(range(len(self.replay_representations_x)), replacement_batch_size)
            for pat in sorted(patterns_to_delete, reverse=True):
                del self.replay_representations_x[pat]
                del self.replay_representations_y[pat]

        # If it will not overfill we just add the new samples
        else:
            x_sample, y_sample = zip(*random.choices(list(zip(train_x, train_y)), k=replacement_batch_size))
            x_sample = self.feature_extractor.predict(np.array(x_sample))

        # Adding new ones
        for i in range(len(x_sample)):
            self.replay_representations_x.append(x_sample[i])
            self.replay_representations_y.append(y_sample[i])

        gc.collect()
        print("Replay X: ",len(self.replay_representations_x)," Replay Y: ",len(self.replay_representations_y))

    # Different approach where we follow the algorithm as shown in Latent Replay for Real-Time Continual Learning
    # Performs much better than the previous approach
    # We essentially reduce the replacement batch size the further we go into our trianing batches
    # https://arxiv.org/abs/1912.01100.pdf
    def storeRepresentationsNativeRehearsal(self, train_x, train_y, batch_num):

        # The replacement batch size will progressively decrease to keep a balanced
        # contribution from the different training batches. We don't have class balancing
        replacement_batch_size = int(self.replay_buffer / batch_num)

        # print("Replacement Batch Size: ",replacement_batch_size)
        # new_replay_memory_size = len(self.replay_representations_x) + replacement_batch_size
        # print("Current Replay Memory Size: ",len(self.replay_representations_x))

        # A check to see if the replacement batch size is greater than the number of samples in a batch
        # This will happen in the first few training batches and depending on how big the replay buffer is
        if replacement_batch_size > len(train_x):
            replacement_batch_size = len(train_x)

        # If the replay buffer will overfill we need to remove some old samples
        # Bare in mind with current implementation, the buffer might end up saving a little bit more than the shown replay buffer size (it's not an issue)
        if batch_num != 1 and (len(self.replay_representations_x) + replacement_batch_size) >= self.replay_buffer:

            x_sample, y_sample = zip(*random.choices(list(zip(train_x, train_y)), k=replacement_batch_size))
            x_sample = self.feature_extractor.predict(np.array(x_sample))

            # Removing old samples
            patterns_to_delete = random.sample(range(len(self.replay_representations_x)), replacement_batch_size)
            for pat in sorted(patterns_to_delete, reverse=True):
                del self.replay_representations_x[pat]
                del self.replay_representations_y[pat]

        # If replay buffer will not overfill we just add the new samples
        else:
            x_sample, y_sample = zip(*random.choices(list(zip(train_x, train_y)), k=replacement_batch_size))
            x_sample = self.feature_extractor.predict(np.array(x_sample))

        # Adding new samples
        for i in range(len(x_sample)):
            self.replay_representations_x.append(x_sample[i])
            self.replay_representations_y.append(y_sample[i])

        # Essential to call gc otherwise we get out of memory errors
        gc.collect()
        print("Replay X: ",len(self.replay_representations_x)," Replay Y: ",len(self.replay_representations_y))

        # Print the number of samples of each class in the replay buffer
        counter_dict = Counter(self.replay_representations_y)
        sorted_counter_dict = {k: counter_dict[k] for k in sorted(counter_dict)}
        print("Replay Buffer Class Distribution: ", sorted_counter_dict)

    # Balanced Reservoir Sampling
    # A random sample from the most represented class is discarded when the Memory Buffer is full
    # https://arxiv.org/abs/2010.05595
    def BRS(self, features, train_x, train_y):

        # features is train_x passed through the feature extractor

        # We take 1 sample at a time
        for i in range(len(train_x)):

            # If replay buffer is not full we just add the new sample
            if self.replay_buffer > len(self.replay_representations_x):
                self.replay_representations_x.append(features[i])
                self.replay_representations_y.append(train_y[i])

            else:

                # Find the class that is most represented in the buffer
                class_counts = np.bincount(self.replay_representations_y)
                most_represented_class = np.argmax(class_counts)

                # Select a random example of the most represented class
                indices = [i for i, y in enumerate(self.replay_representations_y) if y == most_represented_class]
                k = random.choice(indices)

                self.replay_representations_x[k] = features[i]
                self.replay_representations_y[k] = train_y[i] 
        
        gc.collect()
        print("Replay X: ",len(self.replay_representations_x)," Replay Y: ",len(self.replay_representations_y))

        # Print the number of samples of each class in the replay buffer
        counter_dict = Counter(self.replay_representations_y)
        sorted_counter_dict = {k: counter_dict[k] for k in sorted(counter_dict)}
        print("Replay Buffer Class Distribution: ", sorted_counter_dict)

    # Unsure how to exactly get Sbalance and Sloss as score vectors
    # Loss-Aware Balanced Reservoir Sampling
    def LARS(self, features, train_x, train_y, losses):

        # Take 1 sample at a time
        for i in range(len(train_x)):

            # We also store the corresponding loss value
            # sample = (features[i], train_y[i], losses[i])

            if self.replay_buffer > len(self.replay_representations_x):

                # Just add the new sample if replay buffer is not full
                self.replay_representations_x.append(features[i])
                self.replay_representations_y.append(train_y[i])
                self.replay_representations_losses.append(losses[i])

            else:

                # If replay buffer is full, we decide which sample to replace based on their scores

                # Compute Sbalance - occurances of each class
                Sbalance = np.bincount(self.replay_representations_y)
                # Sbalance = class_counts / len(self.replay_representations_x)

                # Compute Sloss
                Sloss = -np.array(self.replay_representations_losses)

                # Combine Sbalance and Sloss to get the final scores
                a = np.sum(np.abs(Sbalance)) / np.sum(np.abs(Sloss))
                S = Sloss * a + Sbalance

                # Compute replacement probabilities
                probs = S / np.sum(S)

                # Choose a sample to replace
                k = np.random.choice(len(self.replay_buffer), p=probs)

                # Replace the chosen sample with the new sample
                self.replay_representations_x[k] = features[i]
                self.replay_representations_y[k] = train_y[i]
                self.replay_representations_losses[k] = losses[i]

        gc.collect()
        print("Replay samples: ", len(self.replay_samples))

        # Print the number of samples of each class in the replay buffer
        counter_dict = Counter(self.replay_representations_y)
        sorted_counter_dict = {k: counter_dict[k] for k in sorted(counter_dict)}
        print("Replay Buffer Class Distribution: ", sorted_counter_dict)

    # Reduntant function since we mix the replay samples with the new samples and train them together in experiments function
    # def replay(self):

    #     replay_x = np.array(self.replay_representations_x)
    #     replay_y = np.array(self.replay_representations_y)

    #     print("> REPLAYING")
    #     # Fitting for just 1 epoch
    #     self.head.fit(replay_x,replay_y,epochs=1,verbose=0) 
