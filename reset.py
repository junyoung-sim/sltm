#!/usr/bin/env python3

import os

for model in os.listdir("./models"):
    if model != ".DS_Store":
        os.system("rm ./models/" + model + "/res/npy/*.npy")
        os.system("rm ./models/" + model + "/res/prediction/*.png")
        os.system("rm ./models/" + model + "/res/validation/*.png")
