import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout
import os


max_path_length = 100
videoDir = 'Videos/'
if os.path.isdir(videoDir)!=True:
    os.makedirs(videoDir , exist_ok = True)

parser = argparse.ArgumentParser()
parser.add_argument('sim_itr', type=int,
                        help='path to the snapshot file')
args = parser.parse_args()

for taskIdx in range(1):

    #_file = 'Task_'+str(taskIdx)+'/itr_'+str(args.sim_itr)+'.pkl'
    _file = 'itr_0.pkl'
    with tf.Session() as sess:

        data = joblib.load(_file)
        policy = data['policy']
        env = data['env']

        uninit_vars = []
        for var in tf.all_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_vars.append(var)
        sess.run(tf.initialize_variables(uninit_vars))

        #policy.std_modifier = 0.00001
        policy.recompute_dist_for_adjusted_std()
       

        path = rollout(env, policy, max_path_length=max_path_length, reset_arg = taskIdx,
                       video_filename=videoDir+'Task'+str(taskIdx)+'.gif' , save_video = True )

    tf.reset_default_graph()
