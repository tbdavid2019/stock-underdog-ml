cd /home/ec2-user/stock-underdog-ml
source /home/ec2-user/stock-underdog-ml/myenv/bin/activate
pip install --upgrade networkx yfinance
python app.py  > /tmp/underdog123.log 2>&1
deactivate

