# Futures

Futures is a deep learning based financial model generating trend forecasts.

## Build and Install
Make sure $PYTHONPATH is set accordingly to each corresponding  system before executing the following commands:
~~~~~~~~~~~~~~~~~~~~~~~
python3 setup.py build
sudo python3 setup.py install
futures.py run <model_name> <symbol>
futures.py train <model_name> <symbol> <start_date> <end_date> <learning_rate> <iteartion> <backtest>
~~~~~~~~~~~~~~~~~~~~~~~

