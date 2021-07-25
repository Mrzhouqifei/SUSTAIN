#！/bin/bash
today=$(date +%Y-%m-%d)
echo "$today" >> log.txt 2>&1
cd /home/zhouqifei/SUSTAIN
chmod +x update_data.py && /home/zhouqifei/anaconda3/bin/python update_data.py >> log.txt 2>&1
chmod +x covid_forecast.py && /home/zhouqifei/anaconda3/bin/python covid_forecast.py >> log.txt 2>&1
chmod +x restructed_data.py && /home/zhouqifei/anaconda3/bin/python restructed_data.py 
cp -rf output/Roland/covid_forecast /home/zhouqifei/SUSTAIN_DATA


cd /home/zhouqifei/SUSTAIN_DATA
git add .
git commit -m "$today"
git push