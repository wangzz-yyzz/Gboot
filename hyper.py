from nni.experiment import Experiment

search_space = {
    "batch_size": {"_type": "choice", "_value": [32]},
    "a": {"_type": "choice", "_value": [0.3, 0.6]},
    "lr": {"_type": "choice", "_value": [0.003]},
    "std": {"_type": "choice", "_value": [0.01, 0.1]},
    "drop": {"_type": "choice", "_value": [0.01, 0.02]},
    "edge_mask": {"_type": "choice", "_value": [0.1, 0.2]},
    "add_history": {"_type": "choice", "_value": [True]},
    "hidden_feature": {"_type": "choice", "_value": [128]},
    "seed": {"_type": "choice", "_value": [1, 2, 3]},
    "task": {"_type": "choice", "_value": ["PEMS08"]},
}

experiment = Experiment('local')
experiment.config.trial_command = 'python train.py'
experiment.config.trial_code_directory = '.'
experiment.config.experiment_working_directory = './log/nni'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
experiment.config.max_trial_number = 200
experiment.config.trial_concurrency = 1
experiment.config.max_trial_duration = '96h'
experiment.run(8080)
