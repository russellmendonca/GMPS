import os
import joblib
import tensorflow as tf
GMPS_PATH = os.environ['GMPS_PATH']
path_to_gmps = GMPS_PATH
test_dir = path_to_gmps + '/seq_test/'
meta_log_dir = test_dir + '/meta_data/'
EXPERT_DATA_LOC = test_dir + '/seq_expert_traj/'

exploc2 = path_to_gmps + '/saved_expert_trajs/ant-quat-v2-10tasks-itr400/'
comp1 = path_to_gmps + "/russel.txt"
comp2 = path_to_gmps + "/zzw.txt"
# with tf.Session().as_default as sess:
#     expert_traces = {taskidx : joblib.load(EXPERT_DATA_LOC+str(taskidx)+".pkl") for taskidx in range(4)}
#     # for taskidx in range(4):
#     #     print(expert_traces[taskidx])
#     file1 = open(comp1, 'w')
#     file2 = open(comp2, 'w')
#     file1.write(str(expert_traces[0]))
#     file2.write(str(expert_traces[3]))
#     file1.close()
#     file2.close()
#
# sess.close()
# expert_traces = {}
# for taskidx in range(4):
#     expert_traces[taskidx] = joblib.load(EXPERT_DATA_LOC+str(taskidx)+".pkl")
#     print("LOADED ", taskidx)
# expert_traces = {taskidx : joblib.load(EXPERT_DATA_LOC+str(taskidx)+".pkl") for taskidx in range(4)}
# print(expert_traces[0])

param = joblib.load(EXPERT_DATA_LOC + "0.pkl")
file1 = open(comp1, 'w')
file1.write(str(param))
