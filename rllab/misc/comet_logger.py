from comet_ml import Experiment, ExistingExperiment
import comet_ml


def get_exp(api_key, previous_experiment_key):
    
    return ExistingExperiment(api_key=api_key,
  previous_experiment=previous_experiment_key)



class CometLogger(Experiment):
    def __init__(self, api_key, project_name, workspace):
        super(CometLogger, self).__init__(api_key, project_name, workspace,
                                          auto_output_logging="simple")
        self.step = 0

    def increase_step(self):
        self.set_step(self.step)
        self.step += 1

    
