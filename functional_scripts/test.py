import sys
from rllab.misc.comet_logger import CometLogger
comet_logger = CometLogger(api_key="KWwx7zh6I2uw6oQMkpEo3smu0",
                            project_name="ml4l3", workspace="glenb")
comet_logger.set_name("test seq train")

from functional_scripts.local_test import experiment as rl_experiment

