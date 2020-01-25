"""
    File to run sequential training simulation

    ### Example of how to run
    GMPS_PATH=/home/gberseth/playground/GMPS MULTIWORLD_PATH=/home/gberseth/playground/multiworld/ python3 functional_scripts/seq_train.py
"""

import sys
import os

GMPS_PATH = os.environ['GMPS_PATH']
MULTIWORL_PATH = os.environ['MULTIWORLD_PATH']
from rllab.misc.comet_logger import CometLogger

comet_logger = CometLogger(api_key="KWwx7zh6I2uw6oQMkpEo3smu0",
                           project_name="ml4l3", workspace="glenb")
comet_logger.set_name("test seq train")

print(comet_logger.get_key())
comet_exp_key = comet_logger.get_key()
# comet_logger.end()

import tensorflow as tf
from functional_scripts.remote_train import experiment as train_experiment
from functional_scripts.local_test import experiment as rl_experiment

path_to_gmps = GMPS_PATH
test_dir = path_to_gmps + '/seq_test/'
meta_log_dir = test_dir + '/meta_data/'
EXPERT_DATA_LOC = test_dir + '/seq_expert_traj/'


def train_seq(meta_variant, rl_variant, comet_logger=comet_logger):
    from multiprocessing import Process
    start_ = 3
    end_ = 10
    # rl_iterations = [2, 4, 6, 8]
    outer_iteration = 0
    for i in range(start_, end_):

        annotation = 'debug-' + str(i) + 'tasks-v0/'

        # policyType = 'conv_fcBiasAda'
        load_policy = None
        n_meta_itr = meta_variant['n_itr']
        if (i > start_):
            load_policy = meta_log_dir + 'debug-' + str(i - 1) + 'tasks-v0/params.pkl'

        meta_variant['log_dir'] = meta_log_dir + annotation
        meta_variant['mbs'] = i
        meta_variant['seed'] = i
        meta_variant['load_policy'] = load_policy
        meta_variant['comet_exp_key'] = comet_exp_key
        meta_variant['outer_iteration'] = outer_iteration
        outer_iteration += meta_variant['n_itr']
        ### fbs is the number of epochs to sample
        ### mbs is the number of tasks to sample using range(0,mbs), so they are not sampled from the full set of tasks.

        # load_policy = '/home/russell/data/s3/Ant-dense-quat-v2-itr400/mri_rosen/policyType_fullAda_Bias/'+\
        #             'ldim_4/adamSteps_500_mbs_40_fbs_50_initFlr_0.5_seed_1/itr_9.pkl'
        # load_policy = '/home/russell/gmps/data/Ant_repl/rep-10tasks-v2/itr_1.pkl'
        # 'imgObs-Sawyer-Push-v4-mpl-50-numDemos5/Itr_250/'

        n_itr = 1
        rl_variant['init_file'] = meta_variant['log_dir'] + '/params.pkl'
        rl_variant['taskIndex'] = i
        rl_variant['n_itr'] = n_itr
        rl_variant['log_dir'] = EXPERT_DATA_LOC
        rl_variant['outer_iteration'] = outer_iteration
        rl_variant['comet_exp_key'] = comet_exp_key
        outer_iteration +=  rl_variant['n_itr']

        if (False):
            proc = Process(target=train_experiment, args=(meta_variant, comet_exp_key))
            proc.start()
            proc.join()
        else:
            train_experiment(variant=meta_variant, comet_exp_key=comet_exp_key)
            tf.reset_default_graph()
            rl_experiment(variant=rl_variant, comet_exp_key=comet_exp_key)
            tf.reset_default_graph()

        # tf.reset_default_graph()

        ## run rl test if necessary
        ## we have trained on tasks 0 ~ i-1, now should test rl on task i
        """
        if i in rl_iterations: ### Glen TODO I am not sure why this is done only specific iterations.
            expPrefix_numItr = expPrefix + '/Task_' + str(i) + '/'
            # for n_itr in range(1,6):
            n_itr = 1
            expName = expPrefix_numItr + 'Itr_' + str(n_itr)
            rl_variant['init_file'] = meta_variant['log_dir'] + '/itr_' + str(n_meta_itr - 1) + '.pkl'
            rl_variant['taskIndex'] = i
            rl_variant['n_itr'] = n_itr
            rl_variant['log_dir'] = RL_OUTPUT_DIR + expName + '/'
            rl_experiment(rl_variant, comet_logger=comet_logger)
            proc = Process(target=rl_experiment, args=(rl_variant, comet_logger.get_key()))
            proc.start()
            proc.join()
            # tf.reset_default_graph()
        """


if __name__ == '__main__':
    path_to_gmps = GMPS_PATH
    path_to_multiworld = MULTIWORL_PATH
    # log_dir = path_to_gmps + '/data/Ant_repl/'
    meta_variant = {'policyType': 'fullAda_Bias',
                    'ldim': 4,
                    'init_flr': 0.5,
                    'seed': None,
                    'log_dir': None,
                    'n_parallel': 4,
                    'envType': 'Ant',
                    'fbs': 10,
                    'mbs': None,
                    'max_path_length': 200,
                    'tasksFile': 'rad2_quat_v2',
                    'load_policy': None,
                    'adam_steps': 500,
                    'dagger': None,
                    'expert_policy_loc': None,
                    'use_maesn': False,
                    # 'expertDataLoc': EXPERT_DATA_LOC,
                    'expertDataLoc': path_to_gmps + '/saved_expert_trajs/ant-quat-v2-10tasks-itr400/',
                    'n_itr': 1}

    ############# RL SETTING ############
    expPrefix = 'Test/Ant/'
    policyType = 'fullAda_Bias'
    if 'conv' in policyType:
        expPrefix = 'img-' + expPrefix

    rl_variant = {'taskIndex': None,
                  'init_file': None,
                  'n_parallel': 4,
                  'log_dir': None,
                  'seed': 1,
                  'tasksFile': 'rad2_quat_v2',
                  'batch_size': 10000,
                  'policyType': policyType,
                  'n_itr': None,
                  'default_step': 0.5,
                  'envType': 'Ant',
                  'max_path_length': 200}

    train_seq(meta_variant=meta_variant, rl_variant=rl_variant, comet_logger=comet_logger)
