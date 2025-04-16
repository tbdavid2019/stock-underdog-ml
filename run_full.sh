
cd /home/ec2-user/stock-underdog-ml
source /home/ec2-user/stock-underdog-ml/myenv/bin/activate
pip install "botocore>=1.37.0"
pip install "urllib3>=2.0.0"
pip install gradio
pip install autogluon.timeseries
pip install -r requirements.txt
python app.py  > /tmp/underdog123.log 2>&1
deactivate

