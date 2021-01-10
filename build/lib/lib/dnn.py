#!/usr/bin/env python3

import random
import logging
import os, ast, json
import numpy as np
import tensorflow.compat.v1 as tf

from .util import write, read

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
tf.disable_v2_behavior()

class DeepNeuralNetwork:
    def __init__(self, model_path="", model_configuration={}):
        """ model_configuration_dict = {"layer_config", "activation", "abs_synapse", "cost", "learning_rate"} """
        self.model_path = model_path + "decoder/"
        if self.load() is False:
            # apply configuration specified in argument if model does not exist
            # ignores configuration specified in argument when model already exists
            self.model_config = model_configuration
        
        layer_config = self.model_config.get("layer_config")
        activation = self.model_config.get("activation")
        abs_synapse = self.model_config.get("abs_synapse")
        cost = self.model_config.get("cost")
        learning_rate = self.model_config.get("learning_rate")
        
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.input, self.output = tf.placeholder(tf.float32), tf.placeholder(tf.float32)
        self.layer = []
        self.original_weights = []
        self.weight = [tf.Variable(tf.random_uniform([l[0], l[1]], -abs_synapse, abs_synapse)) for l in layer_config]
        for l in range(len(layer_config)):
            if l == 0: # input layer
                if activation == "relu":
                    self.layer.append(tf.nn.relu(tf.matmul(self.input, self.weight[l])))
                elif activation == "sigmoid":
                    self.layer.append(tf.nn.sigmoid(tf.matmul(self.input, self.weight[l])))
                elif activation == "tanh":
                    self.layer.append(tf.nn.tanh(tf.matmul(self.input, self.weight[l])))
            else:
                if activation == "relu":
                    self.layer.append(tf.nn.relu(tf.matmul(self.layer[l-1], self.weight[l])))
                elif activation == "sigmoid":
                    self.layer.append(tf.nn.sigmoid(tf.matmul(self.layer[l-1], self.weight[l])))
                elif activation == "tanh":
                    self.layer.append(tf.nn.tanh(tf.matmul(self.layer[l-1], self.weight[l])))
        self.cost = None
        if cost == "MSE":
            self.cost = tf.reduce_mean(tf.squared_difference(self.output, self.layer[-1]))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.model = self.optimizer.minimize(self.cost, global_step=self.global_step)

        self.sess = tf.Session()
        self.saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())
    def save(self):
        self.saver.save(self.sess, self.model_path + "checkpoint.ckpt", global_step=self.global_step)
        with open(self.model_path + "dnn-config", "w+") as File:
            File.write(json.dumps(self.model_config))
    def load(self):
        try:
            with open(self.model_path + "dnn-config", "r") as File:
                self.model_config = ast.literal_eval(File.read())
        except FileNotFoundError:
            return False
    def train(self, training_input=[], training_output=[], epoch=10000, test=0.00):
        try:
            if test != 0.00:
                test_input = np.array([data for data in training_input[-int(training_input.shape[0] * test):]])
                test_output = np.array([data for data in training_output[-int(training_output.shape[0] * test):]])
                training_input = training_input[:-int(training_input.shape[0] * test)]
                training_output = training_output[:-int(training_output.shape[0] * test)]
            for i in range(epoch):
                self.sess.run(self.model, feed_dict={self.input: training_input, self.output: training_output})
                mse = self.sess.run(self.cost, feed_dict={self.input: training_input, self.output: training_output})
                if i % (epoch / 10) == 0:
                    print("EPOCH #{}: COST = {}" .format(i, mse))
            self.save()
            if test != 0.00:
                print("Test Cost: ", self.sess.run(self.cost, feed_dict={self.input: test_input, self.output: test_output}))
                test_result = self.sess.run(self.layer[-1], feed_dict={self.input: test_input})
                return test_result
        except Exception as e:
            print("An error occurred while training model!:\n{}" .format(e))
            return False
    def predict(self, data):
        try:
            prediction = []
            prediction = self.sess.run(self.layer[-1], feed_dict={self.input: data})
            return prediction
        except Exception as e:
            print("An error occurred while running model predictions!:\n{}" .format(e))
    
