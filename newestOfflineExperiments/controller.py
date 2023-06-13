from experiments import Experiments
import argparse
import numpy as np
import tensorflow as tf

# Set seeds for reproducibility
# https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
# Due to the fact that we are running the experiments on GPU, there is some non-determinism involved.
SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_RBS_3000", action="store_true")
    parser.add_argument("--exp_RBS_5000", action="store_true")
    parser.add_argument("--exp_RBS_7500", action="store_true")
    parser.add_argument("--exp_RBS_10000", action="store_true")
    parser.add_argument("--exp_RBS_15000", action="store_true")
    parser.add_argument("--exp_RBS_20000", action="store_true")
    parser.add_argument("--exp_RBS_30000", action="store_true")
    args = parser.parse_args()
    experiments = Experiments()

    # Experimenting with different replay buffer sizes.
    # For each experiment we move the replay layer 1 hidden layer backwards (We move hidden layers from MobileNetV2 to the head of the model and make them trainable)

    if args.exp_RBS_3000:
        print("> Experiment: RBS_3000")

        experiments.runHiddenLayersExperiment(experiment_name="RBS_3000_HL_EXPERIMENTS",
                                                    usecase="RBS_3000_HL0",
                                                    replay_size=3000, num_hidden_layers=0)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_3000_HL_EXPERIMENTS",
                                                    usecase="RBS_3000_HL1",
                                                    replay_size=3000, num_hidden_layers=1)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_3000_HL_EXPERIMENTS",
                                                    usecase="RBS_3000_HL2",
                                                    replay_size=3000, num_hidden_layers=2)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_3000_HL_EXPERIMENTS",
                                                    usecase="RBS_3000_HL3",
                                                    replay_size=3000, num_hidden_layers=3)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_3000_HL_EXPERIMENTS",
                                                    usecase="RBS_3000_HL4",
                                                    replay_size=3000, num_hidden_layers=4)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_3000_HL_EXPERIMENTS",    
                                                    usecase="RBS_3000_HL5",
                                                    replay_size=3000, num_hidden_layers=5)

        experiments.plotExperiment(experiment_name="RBS_3000_HL_EXPERIMENTS",
                                   title="RBS_3000_HL_EXPERIMENTS (CORe50 NICv2 - 391)")
        
    elif args.exp_RBS_5000:
        print("> Experiment: RBS_5000")

        experiments.runHiddenLayersExperiment(experiment_name="RBS_5000_HL_EXPERIMENTS",
                                                    usecase="RBS_5000_HL0",
                                                    replay_size=5000, num_hidden_layers=0)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_5000_HL_EXPERIMENTS",
                                                    usecase="RBS_5000_HL1",
                                                    replay_size=5000, num_hidden_layers=1)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_5000_HL_EXPERIMENTS",
                                                    usecase="RBS_5000_HL2",
                                                    replay_size=5000, num_hidden_layers=2)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_5000_HL_EXPERIMENTS",
                                                    usecase="RBS_5000_HL3",
                                                    replay_size=5000, num_hidden_layers=3)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_5000_HL_EXPERIMENTS",
                                                    usecase="RBS_5000_HL4",
                                                    replay_size=5000, num_hidden_layers=4)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_5000_HL_EXPERIMENTS",
                                                    usecase="RBS_5000_HL5",
                                                    replay_size=5000, num_hidden_layers=5)

        experiments.plotExperiment(experiment_name="RBS_5000_HL_EXPERIMENTS",
                                   title="RBS_5000_HL_EXPERIMENTS (CORe50 NICv2 - 391)")

    elif args.exp_RBS_7500:
        print("> Experiment: RBS_7500")

        experiments.runHiddenLayersExperiment(experiment_name="RBS_7500_HL_EXPERIMENTS",
                                                    usecase="RBS_7500_HL0",
                                                    replay_size=7500, num_hidden_layers=0)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_7500_HL_EXPERIMENTS",
                                                    usecase="RBS_7500_HL1",
                                                    replay_size=7500, num_hidden_layers=1)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_7500_HL_EXPERIMENTS",
                                                    usecase="RBS_7500_HL2",
                                                    replay_size=7500, num_hidden_layers=2)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_7500_HL_EXPERIMENTS",
                                                    usecase="RBS_7500_HL3",
                                                    replay_size=7500, num_hidden_layers=3)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_7500_HL_EXPERIMENTS",
                                                    usecase="RBS_7500_HL4",
                                                    replay_size=7500, num_hidden_layers=4)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_7500_HL_EXPERIMENTS",
                                                    usecase="RBS_7500_HL5",
                                                    replay_size=7500, num_hidden_layers=5)
        
        experiments.plotExperiment(experiment_name="RBS_7500_HL_EXPERIMENTS",
                                   title="RBS_7500_HL_EXPERIMENTS (CORe50 NICv2 - 391)")

    elif args.exp_RBS_10000:
        print("> Experiment: RBS_10000")

        experiments.runHiddenLayersExperiment(experiment_name="RBS_10000_HL_EXPERIMENTS",
                                                    usecase="RBS_10000_HL0",
                                                    replay_size=10000, num_hidden_layers=0)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_10000_HL_EXPERIMENTS",
                                                    usecase="RBS_10000_HL1",
                                                    replay_size=10000, num_hidden_layers=1)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_10000_HL_EXPERIMENTS",
                                                    usecase="RBS_10000_HL2",
                                                    replay_size=10000, num_hidden_layers=2)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_10000_HL_EXPERIMENTS",
                                                    usecase="RBS_10000_HL3",
                                                    replay_size=10000, num_hidden_layers=3)

        experiments.runHiddenLayersExperiment(experiment_name="RBS_10000_HL_EXPERIMENTS",
                                                    usecase="RBS_10000_HL4",
                                                    replay_size=10000, num_hidden_layers=4)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_10000_HL_EXPERIMENTS",
                                                    usecase="RBS_10000_HL5",
                                                    replay_size=10000, num_hidden_layers=5)
        
        experiments.plotExperiment(experiment_name="RBS_10000_HL_EXPERIMENTS",
                                    title="RBS_10000_HL_EXPERIMENTS (CORe50 NICv2 - 391)")

    elif args.exp_RBS_15000:
        print("> Experiment: RBS_15000")

        experiments.runHiddenLayersExperiment(experiment_name="RBS_15000_HL_EXPERIMENTS",
                                                    usecase="RBS_15000_HL0",
                                                    replay_size=15000, num_hidden_layers=0)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_15000_HL_EXPERIMENTS",
                                                    usecase="RBS_15000_HL1",
                                                    replay_size=15000, num_hidden_layers=1)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_15000_HL_EXPERIMENTS",
                                                    usecase="RBS_15000_HL2",
                                                    replay_size=15000, num_hidden_layers=2)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_15000_HL_EXPERIMENTS",
                                                    usecase="RBS_15000_HL3",
                                                    replay_size=15000, num_hidden_layers=3)

        experiments.runHiddenLayersExperiment(experiment_name="RBS_15000_HL_EXPERIMENTS",
                                                    usecase="RBS_15000_HL4",
                                                    replay_size=15000, num_hidden_layers=4)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_15000_HL_EXPERIMENTS",
                                                    usecase="RBS_15000_HL5",
                                                    replay_size=15000, num_hidden_layers=5)
        
        experiments.plotExperiment(experiment_name="RBS_15000_HL_EXPERIMENTS",
                                    title="RBS_15000_HL_EXPERIMENTS (CORe50 NICv2 - 391)")

    elif args.exp_RBS_20000:
        print("> Experiment: RBS_20000")

        experiments.runHiddenLayersExperiment(experiment_name="RBS_20000_HL_EXPERIMENTS",
                                                    usecase="RBS_20000_HL0",
                                                    replay_size=20000, num_hidden_layers=0)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_20000_HL_EXPERIMENTS",
                                                    usecase="RBS_20000_HL1",
                                                    replay_size=20000, num_hidden_layers=1)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_20000_HL_EXPERIMENTS",
                                                    usecase="RBS_20000_HL2",
                                                    replay_size=20000, num_hidden_layers=2)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_20000_HL_EXPERIMENTS",
                                                    usecase="RBS_20000_HL3",
                                                    replay_size=20000, num_hidden_layers=3)

        experiments.runHiddenLayersExperiment(experiment_name="RBS_20000_HL_EXPERIMENTS",
                                                    usecase="RBS_20000_HL4",
                                                    replay_size=20000, num_hidden_layers=4)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_20000_HL_EXPERIMENTS",
                                                    usecase="RBS_20000_HL5",
                                                    replay_size=20000, num_hidden_layers=5)
        
        experiments.plotExperiment(experiment_name="RBS_20000_HL_EXPERIMENTS",
                                    title="RBS_20000_HL_EXPERIMENTS (CORe50 NICv2 - 391)")

    elif args.exp_RBS_30000:
        print("> Experiment: RBS_30000")

        experiments.runHiddenLayersExperiment(experiment_name="RBS_30000_HL_EXPERIMENTS",
                                                    usecase="RBS_30000_HL0",
                                                    replay_size=30000, num_hidden_layers=0)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_30000_HL_EXPERIMENTS",
                                                    usecase="RBS_30000_HL1",
                                                    replay_size=30000, num_hidden_layers=1)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_30000_HL_EXPERIMENTS",
                                                    usecase="RBS_30000_HL2",
                                                    replay_size=30000, num_hidden_layers=2)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_30000_HL_EXPERIMENTS",
                                                    usecase="RBS_30000_HL3",
                                                    replay_size=30000, num_hidden_layers=3)

        experiments.runHiddenLayersExperiment(experiment_name="RBS_30000_HL_EXPERIMENTS",
                                                    usecase="RBS_30000_HL4",
                                                    replay_size=30000, num_hidden_layers=4)
        
        experiments.runHiddenLayersExperiment(experiment_name="RBS_30000_HL_EXPERIMENTS",
                                                    usecase="RBS_30000_HL5",
                                                    replay_size=30000, num_hidden_layers=5)
        
        experiments.plotExperiment(experiment_name="RBS_30000_HL_EXPERIMENTS",
                                    title="RBS_30000_HL_EXPERIMENTS (CORe50 NICv2 - 391)")    

    else:
        print("> No valid experiment option provided")