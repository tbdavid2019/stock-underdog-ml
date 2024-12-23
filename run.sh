
cd /home/ubuntu/underdog
source /home/ubuntu/underdog/myenv/bin/activate
pip install -r requirements.txt
python app.py  > /tmp/underdog123.log 2>&1
deactivate

