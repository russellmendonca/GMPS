from comet_ml import Experiment, ExistingExperiment


def get_exp(api_key, previous_experiment_key):
    
    return ExistingExperiment(api_key=api_key, previous_experiment=previous_experiment_key)


class CometLogger(Experiment):
    def __init__(self, api_key=None, project_name=None, workspace=None, previous_experiment_key=None):
        if (previous_experiment_key is not None):
            self = ExistingExperiment(api_key=api_key, previous_experiment=previous_experiment_key)
        else:
            super(CometLogger, self).__init__(api_key, project_name, workspace,
                                          auto_output_logging="simple")
        self._step = 0

