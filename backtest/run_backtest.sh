#!/bin/bash
# Backtest automation script
# Run this weekly to update actual prices for past predictions

cd /home/ec2-user/stock-underdog-ml
source myenv/bin/activate
python backtest/backtest.py >> logs/backtest.log 2>&1
