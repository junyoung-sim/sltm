# Futures

Futures is a deep learning based financial model generating trend forecasts.

## Build and Install
Install required packages using pip:
~~~~~~~~~~~~~~~~~~~~~~
pip3 install -r requirements.txt
~~~~~~~~~~~~~~~~~~~~~~

Build a standalone executable
~~~~~~~~~~~~~~~~~~~~~~
pyinstaller --onefile futures.py
~~~~~~~~~~~~~~~~~~~~~~

## Application
Run these following commands at any location to use Futures.
All data produced, including the models and results, will be saved where the command was executed.
Refer to futures.py for further use instructions
~~~~~~~~~~~~~~~~~~~~~~~
./futures run <model_name> <symbol>
./futures train <model_name> <symbol> <start_date> <end_date> <learning_rate> <iteartion> <backtest>
~~~~~~~~~~~~~~~~~~~~~~~

