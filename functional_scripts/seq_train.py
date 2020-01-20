import sys
sys.path.append("../R_multiworld")
from rllab.misc.comet_logger import CometLogger
comet_logger = CometLogger(api_key="KWwx7zh6I2uw6oQMkpEo3smu0",
                            project_name="ml4l3", workspace="glenb")
comet_logger.set_name("test seq train")

import tensorflow as tf
from functional_scripts.remote_train import experiment as train_experiment
from functional_scripts.local_test import experiment as rl_experiment

user = 'root'
path_to_gmps = '/' + str(user) + '/playground/GMPS/'
meta_log_dir = path_to_gmps + '/data/seq_test/meta_data/'
RL_OUTPUT_DIR = path_to_gmps + '/data/seq_test/rl_data/'


def main(meta_variant, rl_variant, comet_logger=comet_logger):

    start_ = 3
    end_ = 10
    rl_iterations = [2, 4, 6, 8]
    for i in range(start_, end_):

        annotation = 'debug-' + str(i) + 'tasks-v0/'

        # policyType = 'conv_fcBiasAda'
        load_policy = None
        n_meta_itr = meta_variant['n_itr']
        if (i > start_):
            load_policy = meta_log_dir + 'debug-' + str(i - 1) + 'tasks-v0/itr_' + str(n_meta_itr - 1) + '.pkl'

        meta_variant['log_dir'] = meta_log_dir + annotation
        meta_variant['mbs'] = i
        meta_variant['seed'] = i
        meta_variant['load_policy'] = load_policy
        ### fbs is the number of epochs to sample
        ### mbs is the number of tasks to sample using range(0,mbs), so they are not sampled from the full set of tasks.

        # load_policy = '/home/russell/data/s3/Ant-dense-quat-v2-itr400/mri_rosen/policyType_fullAda_Bias/'+\
        #             'ldim_4/adamSteps_500_mbs_40_fbs_50_initFlr_0.5_seed_1/itr_9.pkl'
        # load_policy = '/home/russell/gmps/data/Ant_repl/rep-10tasks-v2/itr_1.pkl'
        # 'imgObs-Sawyer-Push-v4-mpl-50-numDemos5/Itr_250/'
        train_experiment(variant=meta_variant, comet_logger=comet_logger)
        tf.reset_default_graph()

        ## run rl test if necessary
        ## we have trained on tasks 0 ~ i-1, now should test rl on task i
        if i in rl_iterations:
            expPrefix_numItr = expPrefix + '/Task_' + str(i) + '/'
            # for n_itr in range(1,6):
            n_itr = 1
            expName = expPrefix_numItr + 'Itr_' + str(n_itr)
            rl_variant['init_file'] = meta_variant['log_dir'] + '/itr_' + str(n_meta_itr - 1) + '.pkl'
            rl_variant['taskIndex'] = i
            rl_variant['n_itr'] = n_itr
            rl_variant['log_dir'] = RL_OUTPUT_DIR + expName + '/'
            rl_experiment(rl_variant, comet_logger=comet_logger)
            tf.reset_default_graph()


if __name__ == '__main__':
    user = 'root'
    path_to_gmps = '/' + str(user) + '/playground/GMPS/'
    path_to_multiworld = '/' + str(user) + '/playground/R_multiworld/'
    # log_dir = path_to_gmps + '/data/Ant_repl/'
    meta_variant = {'policyType': 'fullAda_Bias',
               'ldim': 4,
               'init_flr': 0.5,
               'seed': None,
               'log_dir': None,
               'n_parallel': 1,
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
               'expertDataLoc': path_to_gmps + '/saved_expert_trajs/ant-quat-v2-10tasks-itr400/',
               'n_itr': 6}

    ############# RL SETTING ############
    expPrefix = 'Test/Ant/'
    policyType = 'fullAda_Bias'
    if 'conv' in policyType:
        expPrefix = 'img-' + expPrefix

    rl_variant = {'taskIndex': None,
               'init_file': None,
               'n_parallel': 1,
               'log_dir': None,
               'seed': 1,
               'tasksFile': 'rad2_quat_v2',
               'batch_size': 10000,
               'policyType': policyType,
               'n_itr': None,
               'default_step': 0.5,
               'envType': 'Ant',
               'max_path_length': 200}

    main(meta_variant=meta_variant, rl_variant=rl_variant, comet_logger=comet_logger)
