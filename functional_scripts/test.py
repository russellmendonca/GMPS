import os
import joblib
GMPS_PATH = os.environ['GMPS_PATH']
path_to_gmps = GMPS_PATH
test_dir = path_to_gmps + '/seq_test/'
meta_log_dir = test_dir + '/meta_data/'
EXPERT_DATA_LOC = test_dir + '/seq_expert_traj/'

exploc2 = path_to_gmps + '/saved_expert_trajs/ant-quat-v2-10tasks-itr400/'

expert_traces = {taskidx : joblib.load(EXPERT_DATA_LOC+str(taskidx)+".pkl") for taskidx in range(3)}
# for taskidx in range(4):
#     print(expert_traces[taskidx])
print(expert_traces[0])
print(expert_traces[1])

# expert_traces2 = {taskidx : joblib.load(EXPERT_DATA_LOC+str(taskidx)+".pkl") for taskidx in range(3)}
#
# for taskidx in range(3):
#     print(expert_traces2[taskidx])
