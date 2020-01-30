import sys

from rllab.misc.comet_logger import CometLogger
# comet_logger = CometLogger(api_key="KWwx7zh6I2uw6oQMkpEo3smu0",
#                             project_name="ml4l3", workspace="glenb")
# comet_logger.set_name("local_test rl")
comet_logger=None

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from comet_ml import Experiment, ExistingExperiment

from sandbox.rocky.tf.algos.vpg import VPG as vpg_basic
from sandbox.rocky.tf.algos.vpg_biasADA import VPG as vpg_biasADA
from sandbox.rocky.tf.algos.vpg_fullADA import VPG as vpg_fullADA
from sandbox.rocky.tf.algos.vpg_conv import VPG as vpg_conv

# from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy_adaptivestep_biastransform import MAMLGaussianMLPPolicy as fullAda_Bias_policy
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy_biasonlyadaptivestep_biastransform import \
    MAMLGaussianMLPPolicy as biasAda_Bias_policy

from multiworld.envs.mujoco.sawyer_xyz.push.sawyer_push import SawyerPushEnv
from multiworld.envs.mujoco.sawyer_xyz.pickPlace.sawyer_pick_and_place import SawyerPickPlaceEnv
from multiworld.envs.mujoco.sawyer_xyz.door.sawyer_door_open import SawyerDoorOpenEnv
from multiworld.envs.mujoco.sawyer_xyz.multi_domain.push_door import Sawyer_MultiDomainEnv
from multiworld.envs.mujoco.sawyer_xyz.pickPlace.sawyer_coffee import SawyerCoffeeEnv

from rllab.envs.mujoco.ant_env_rand_goal_ring import AntEnvRandGoalRing
from multiworld.core.flat_goal_env import FlatGoalEnv
from multiworld.core.finn_maml_env import FinnMamlEnv
from multiworld.core.wrapper_env import NormalizedBoxEnv
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

import pickle
import argparse
from sandbox.rocky.tf.envs.base import TfEnv

import csv
import joblib
import numpy as np
import pickle
import tensorflow as tf
import joblib
# import doodad as dd
# from doodad.exp_utils import setup
import rllab.misc.logger as logger
from rllab.misc.ext import  set_seed
import os

import os
GMPS_PATH = os.environ['GMPS_PATH']
MULTIWORL_PATH= os.environ['MULTIWORLD_PATH']
path_to_gmps = GMPS_PATH
path_to_multiworld = MULTIWORL_PATH
OUTPUT_DIR = path_to_gmps + '/data/local/'


def setup(seed, n_parallel, log_dir):
    if seed is not None:
        set_seed(seed)

    if n_parallel > 0:
        from rllab.sampler import parallel_sampler
        parallel_sampler.initialize(n_parallel=n_parallel)
        if seed is not None:
            parallel_sampler.set_seed(seed)

    if os.path.isdir(log_dir) == False:
        os.makedirs(log_dir, exist_ok=True)

    logger._snapshot_mode = 'last'
    logger.set_snapshot_dir(log_dir)
    logger.add_tabular_output(log_dir + '/progress.csv')


def experiment(variant, comet_exp_key=None):
    if comet_exp_key is not None:
        # from rllab.misc.comet_logger import CometContinuedLogger, CometLogger
        # from comet_ml import Experiment, ExistingExperiment
        # comet_log = CometContinuedLogger(api_key="KWwx7zh6I2uw6oQMkpEo3smu0", previous_experiment_key=variant['comet_exp_key'])
        comet_logger = ExistingExperiment(api_key="KWwx7zh6I2uw6oQMkpEo3smu0",
                                       previous_experiment=variant['comet_exp_key'])
        # comet_log = CometLogger(api_key="KWwx7zh6I2uw6oQMkpEo3smu0",
        #                     project_name="ml4l3", workspace="glenb")
        comet_logger.set_name("test seq train")
        # comet_log = comet_exp_key
        print("RL!: ", comet_logger)
    print("%%%%%%%%%%%%%%%%%", comet_logger)
    seed = variant['seed']
    log_dir = variant['log_dir']
    n_parallel = variant['n_parallel']


    setup(seed, n_parallel, log_dir)

    init_file = variant['init_file']
    taskIndex = variant['taskIndex']
    n_itr = variant['n_itr']
    default_step = variant['default_step']
    policyType = variant['policyType']
    envType = variant['envType']

    tasksFile = path_to_multiworld + '/multiworld/envs/goals/' + variant['tasksFile'] + '.pkl'
    tasks = pickle.load(open(tasksFile, 'rb'))

    max_path_length = variant['max_path_length']

    use_images = 'conv' in policyType
    print("$$$$$$$$$$$$$$$ RL-TASK: ", str(tasks[taskIndex]), " $$$$$$$$$$$$$$$")
    if 'MultiDomain' in envType:
        baseEnv = Sawyer_MultiDomainEnv(tasks=tasks, image=use_images, mpl=max_path_length)

    elif 'Push' in envType:
        baseEnv = SawyerPushEnv(tasks=tasks, image=use_images, mpl=max_path_length)


    elif 'PickPlace' in envType:
        baseEnv = SawyerPickPlaceEnv(tasks=tasks, image=use_images, mpl=max_path_length)

    elif 'Door' in envType:
        baseEnv = SawyerDoorOpenEnv(tasks=tasks, image=use_images, mpl=max_path_length)

    elif 'Ant' in envType:
        env = TfEnv(normalize(AntEnvRandGoalRing()))

    elif 'Coffee' in envType:
        baseEnv = SawyerCoffeeEnv(mpl=max_path_length)

    else:
        raise AssertionError('')

    if envType in ['Push', 'PickPlace', 'Door']:
        if use_images:
            obs_keys = ['img_observation']
        else:
            obs_keys = ['state_observation']
        env = TfEnv(NormalizedBoxEnv(FinnMamlEnv(FlatGoalEnv(baseEnv, obs_keys=obs_keys), reset_mode='idx')))

    baseline = ZeroBaseline(env_spec=env.spec)
    # baseline = LinearFeatureBaseline(env_spec = env.spec)
    batch_size = variant['batch_size']

    if policyType == 'fullAda_Bias':

        baseline = LinearFeatureBaseline(env_spec=env.spec)
        algo = vpg_fullADA(
            env=env,
            policy=None,
            load_policy=init_file,
            baseline=baseline,
            batch_size=batch_size,  # 2x
            max_path_length=max_path_length,
            n_itr=n_itr,
            # noise_opt = True,
            default_step=default_step,
            sampler_cls=VectorizedSampler,  # added by RK 6/19
            sampler_args=dict(n_envs=1),

            # reset_arg=np.asscalar(taskIndex),
            reset_arg=taskIndex,
            log_dir=log_dir,
            comet_logger=comet_logger,
            outer_iteration=variant['outer_iteration']
        )

    elif policyType == 'biasAda_Bias':

        algo = vpg_biasADA(
            env=env,
            policy=None,
            load_policy=init_file,
            baseline=baseline,
            batch_size=batch_size,  # 2x
            max_path_length=max_path_length,
            n_itr=n_itr,
            # noise_opt = True,
            default_step=default_step,
            sampler_cls=VectorizedSampler,  # added by RK 6/19
            sampler_args=dict(n_envs=1),
            # reset_arg=np.asscalar(taskIndex),
            reset_arg=taskIndex,
            log_dir=log_dir
        )

    elif policyType == 'basic':

        algo = vpg_basic(
            env=env,
            policy=None,
            load_policy=init_file,
            baseline=baseline,
            batch_size=batch_size,
            max_path_length=max_path_length,
            n_itr=n_itr,
            # step_size=10.0,
            sampler_cls=VectorizedSampler,  # added by RK 6/19
            sampler_args=dict(n_envs=1),

            reset_arg=taskIndex,
            optimizer=None,
            optimizer_args={'init_learning_rate': default_step,
                            'tf_optimizer_args': {'learning_rate': 0.5 * default_step},
                            'tf_optimizer_cls': tf.train.GradientDescentOptimizer},
            log_dir=log_dir
            # extra_input="onehot_exploration", # added by RK 6/19
            # extra_input_dim=5, # added by RK 6/19
        )


    elif 'conv' in policyType:

        algo = vpg_conv(
            env=env,
            policy=None,
            load_policy=init_file,
            baseline=baseline,
            batch_size=batch_size,  # 2x
            max_path_length=max_path_length,
            n_itr=n_itr,
            sampler_cls=VectorizedSampler,  # added by RK 6/19
            sampler_args=dict(n_envs=1),
            # noise_opt = True,
            default_step=default_step,
            # reset_arg=np.asscalar(taskIndex),
            reset_arg=taskIndex,
            log_dir=log_dir

        )

    else:
        raise AssertionError('Policy Type must be fullAda_Bias or biasAda_Bias')

    algo.train()


# val = False

####################### Example Testing script for Pushing ####################################
# envType = 'Push' ; max_path_length = 50 ; tasksFile = 'push_v4_val'


# envType = 'Ant';
# annotation = 'v2-40tasks';
# tasksFile = 'rad2_quat_v2';
# max_path_length = 200
# initFile = path_to_gmps + '/data/Ant_repl/' + 'debug-40tasks-v2/' + 'itr_99.pkl'
# policyType = 'biasAda_Bias'
# policyType = 'conv_fcBiasAda'

# initFlr = 0.5;
# seed = 1
# batch_size = 10000

# Provide the meta-trained file which will be used for testing

if __name__ == '__main__':

    expPrefix = 'Test/Ant/'
    policyType = 'fullAda_Bias'
    if 'conv' in policyType:
        expPrefix = 'img-' + expPrefix

    variant = {'taskIndex': 0,
               'init_file': path_to_gmps + '/data/Ant_repl/' + 'debug-40tasks-v2/' + 'itr_99.pkl',
               'n_parallel': 1,
               'log_dir': '',
               'seed': 1,
               'tasksFile': 'rad2_quat_v2',
               'batch_size': 10000,
               'policyType': policyType,
               'n_itr': 0,
               'default_step': 0.5,
               'envType': 'Ant',
               'max_path_length': 200}

    for index in [2]:

        for n_itr in [20]:
            tf.reset_default_graph()
            expPrefix_numItr = expPrefix + '/Task_' + str(index) + '/'

            # for n_itr in range(1,6):

            expName = expPrefix_numItr + 'Itr_' + str(n_itr)
            variant['taskIndex'] = index
            variant['n_itr'] = n_itr
            variant['log_dir'] = OUTPUT_DIR + expName + '/'
            experiment(variant, comet_logger=comet_logger)



