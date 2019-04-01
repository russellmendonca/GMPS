import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('reset_arg', type=int)
    parser.add_argument('--max_path_length', type=int, default=150,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--video_filename', type=str,
                        help='path to the out video file')
    parser.add_argument('--prompt', type=bool, default=False,
                        help='Whether or not to prompt for more sim')
    args = parser.parse_args()

    max_tries = 10
    tri = 0
  
         
    with tf.Session() as sess:

        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env']

        uninit_vars = []
        for var in tf.all_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_vars.append(var)
        sess.run(tf.initialize_variables(uninit_vars))

        path = rollout(env, policy, max_path_length=args.max_path_length, reset_arg = args.reset_arg,
                        speedup=args.speedup, video_filename='sim_out.mp4' , save_video = True )
       