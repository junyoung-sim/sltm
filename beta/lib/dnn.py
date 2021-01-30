
import os, logging, json, ast
import tensorflow.compat.v1 as tf

# suppress warnings from TensorFlow
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
tf.disable_v2_behavior()

class DeepNeuralNetwork:
    def __init__(self, path="", hyper={}):
        """ hyper = {"architecture", "activation", "abs_synapse", "cost", "learning_rate"} """
        self.path = path + "/dnn"
        if self.load() == False: # if model does not exist
            # if the model exists, self.load() automatically loads hyperparameters that are saved
            self.hyper = hyper # load given hyperparameters
        # load hyperparameters
        architecture  = hyper["architecture"]
        activation    = hyper["activation"]
        abs_synapse   = hyper["abs_synpase"]
        cost          = hyper["cost"]
        learning_rate = hyper["learning_rate"]
    def save(self):
        with open(self.path + "/hyperparameters", "w+") as f:
            f.write(json.dumps(self.hyper))
    def load(self):
        try:
            with open(self.path + "/hyperparameters", "r") as f:
                self.hyper = ast.literal_eval(f.read())
        except FileNotFoundError:
            return False
