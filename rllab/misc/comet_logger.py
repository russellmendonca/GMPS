from comet_ml import Experiment

class CometLogger(Experiment):
    def __init__(self, api_key, project_name, workspace):
        super(CometLogger, self).__init__(api_key, project_name, workspace)
        self.step = 0

    def increase_step(self):
        self.set_step(self.step)
        self.step += 1
