import sys
sys.path.append("../R_multiworld")
from rllab.misc.comet_logger import CometLogger
comet_logger = CometLogger(api_key="KWwx7zh6I2uw6oQMkpEo3smu0",
                            project_name="ml4l3", workspace="glenb")
comet_logger.set_name("test save gmps 50 itr")

import tensorflow as tf
from functional_scripts.remote_train import experiment as train_experiment

def main(variant, comet_logger=comet_logger):

    start_ = 3
    end_ = 10
    for i in range(start_, end_):

        user = 'root'
        path_to_gmps = '/' + str(user) + '/playground/GMPS/'
        log_dir = path_to_gmps + '/data/Ant_repl/'
        annotation = 'debug-' + str(i) + 'tasks-v0'

        # policyType = 'conv_fcBiasAda'
        load_policy = None
        if (i > start_):
            load_policy = log_dir + 'debug-' + str(i - 1) + 'tasks-v0/itr_5.pkl'

        variant['lod_dir'] = log_dir + annotation
        variant['seed'] = i
        variant['load_policy'] = load_policy
        ### fbs is the number of epochs to sample
        ### mbs is the number of tasks to sample using range(0,mbs), so they are not sampled from the full set of tasks.

        # load_policy = '/home/russell/data/s3/Ant-dense-quat-v2-itr400/mri_rosen/policyType_fullAda_Bias/'+\
        #             'ldim_4/adamSteps_500_mbs_40_fbs_50_initFlr_0.5_seed_1/itr_9.pkl'
        # load_policy = '/home/russell/gmps/data/Ant_repl/rep-10tasks-v2/itr_1.pkl'
        # 'imgObs-Sawyer-Push-v4-mpl-50-numDemos5/Itr_250/'
        train_experiment(variant=variant, comet_logger=comet_logger)
        tf.reset_default_graph()


if __name__ == '__main__':
    user = 'root'
    path_to_gmps = '/' + str(user) + '/playground/GMPS/'
    path_to_multiworld = '/' + str(user) + '/playground/R_multiworld/'
    log_dir = path_to_gmps + '/data/Ant_repl/'
    annotation = ''
    variant = {'policyType': 'fullAda_Bias',
               'ldim': 4,
               'init_flr': 0.5,
               'seed': 0,
               'log_dir': log_dir + annotation,
               'n_parallel': 1,
               'envType': 'Ant',
               'fbs': 10,
               'mbs': 1,
               'max_path_length': 200,
               'tasksFile': 'rad2_quat_v2',
               'load_policy': None,
               'adam_steps': 500,
               'dagger': None,
               'expert_policy_loc': None,
               'use_maesn': False,
               'expertDataLoc': path_to_gmps + '/saved_expert_trajs/ant-quat-v2-10tasks-itr400/',
               'iterations': 6}
