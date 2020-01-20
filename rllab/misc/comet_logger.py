from comet_ml import Experiment, ExistingExperiment


class CometContinuedLogger(ExistingExperiment):
    
    def __init__(self, api_key=None, previous_experiment_key=None):
        # self = ExistingExperiment(api_key=api_key, previous_experiment=previous_experiment_key)
        super(ExistingExperiment, self).__init__(api_key, previous_experiment_key)

class CometLogger(Experiment):
    def __init__(self, api_key=None, project_name=None, workspace=None, previous_experiment_key=None):
        super(CometLogger, self).__init__(api_key, project_name, workspace)

