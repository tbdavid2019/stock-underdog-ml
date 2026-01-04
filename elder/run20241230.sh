
cd /home/ec2-user/stock-underdog-ml
source /home/ec2-user/stock-underdog-ml/myenv/bin/activate
pip install -r requirements.txt
python app20241230.py  > /tmp/underdog20241230.log 2>&1
deactivate

