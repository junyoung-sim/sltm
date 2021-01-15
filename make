#!/bin/sh

sudo rm -rf build dist futures.egg-info
python3 setup.py build
sudo python3 setup.py install
