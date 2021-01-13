# Futures

Futures is a deep learning based financial model generating trend forecasts.

## Build and Install
Make sure $PYTHONPATH is set properly in ~/.zshrc or any equivalent script. For instance:
~~~~~~~~~~~~~~~~~~~~~~
export PYTHONPATH=/Library/Python/3.8/site-packages/$PYTHONPATH
~~~~~~~~~~~~~~~~~~~~~~

Run the following commands to install Futures:
~~~~~~~~~~~~~~~~~~~~~~~
python3 setup.py build
sudo python3 setup.py install
~~~~~~~~~~~~~~~~~~~~~~~

## Application
Run these following commands at any location to use Futures.
All data produced, including the models and results, will be saved where the command was executed.
~~~~~~~~~~~~~~~~~~~~~~~
futures.py run <model_name> <symbol>
futures.py train <model_name> <symbol> <start_date> <end_date> <learning_rate> <iteartion> <backtest>
~~~~~~~~~~~~~~~~~~~~~~~

