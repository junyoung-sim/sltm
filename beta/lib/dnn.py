
import os, logging
import tensorflow.compat.v1 as tf

# suppress warnings from TensorFlow
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
tf.disable_v2_behavior()

