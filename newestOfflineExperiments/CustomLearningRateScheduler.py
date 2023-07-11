import tensorflow as tf

# Exponential LR Decay
# lr = lr0 * gamma ^ (totalnumsamplesseen)
# gamma is a hyperparameter that we choose. Goal is at the end of training to have a lr equal to 1/6 of the initial lr

class CustomLearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, gamma):
        super().__init__()

        self.initial_learning_rate = initial_learning_rate
        self.gamma = gamma

    # Step gives current batch. Can do * 300 to get total samples so far since we see 300 everytime besides epoch 1
    def __call__(self, step):
        return self.initial_learning_rate * (self.gamma ** (step))

# Can use these

# lr_schedule = CustomLearningRateScheduler(initial_learning_rate=0.001, gamma=0.9999846859337639)
# optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
