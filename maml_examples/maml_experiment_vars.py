import tensorflow as tf

HIDDEN_NONLINEARITY = {
    '':tf.nn.relu,
    'relu':tf.nn.relu,
    'th':tf.nn.tanh,
    'thh':tf.nn.tanh,
    'reluh':tf.nn.relu,
}
OUTPUT_NONLINEARITY = {
    '':tf.identity,
    'relu':tf.identity,
    'th':tf.identity,
    'thh':tf.nn.tanh,
    'reluh':tf.nn.tanh,
}

TESTING_ITRS = [2*x for x in range(10000)] #+ [100*x-1 for x in range(1000)]
# TESTING_ITRS = []
# TESTING_ITRS = [2*x for x in range(1000)]
# TESTING_ITRS = [0,2,3,5,7,9,11,13,15,17,19] + [2*x+1 for x in range(1000)]
# BASELINE_TRAINING_ITRS = [0,]
BASELINE_TRAINING_ITRS = []
assert set(BASELINE_TRAINING_ITRS).issubset(set(TESTING_ITRS))

PLOT_ITRS = [0,2,4,6,8,10] + [100*x for x in range(80)] + [1,3,5,7,9,11,13,] + [8*x-1 for x in range(500)]

VIDEO_ITRS = TESTING_ITRS
# VIDEO_ITRS = TESTING_ITRS
# VIDEO_ITRS = [1,3,5,7,9,11,13,15,17] # [11,13,15,17,19] + [8*x-1 for x in range(2,1000)]
# VIDEO_ITRS = [1,3] + [200*x-1 for x in range(40)]



def clip05_20(x):
    return tf.clip_by_value(x,0.5,2.0)

def clip_wide(x):
    return tf.clip_by_value(x,1e-8,1e8)

def clip05_(x):
    return tf.maximum(x,0.5)
def clip1_(x):
    return tf.maximum(x,1.0)


def clip1_1(x):
    return tf.clip_by_value(x,1.0,1.0)


def clip095_105(x):
    return tf.clip_by_value(x,0.95,1.05)

MOD_FUNC={
    '':tf.identity,
    'ism_sqrt':tf.sqrt,
    'ism_square':tf.square,
    'clip0.5_2.0':clip05_20,
    'clip0.95_1.05':clip095_105,
    'clip0.5_':clip05_,
    'clip1.0_':clip1_,
    'clip1.0_1.0':clip1_1,
    'clipX':clip_wide, # fwiw clipX seems to do the same
}
