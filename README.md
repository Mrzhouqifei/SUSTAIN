# SUSTAIN Model Run Instructions
The Social and Sustainable Artificial Intelligence (SUSTAIN) Model for Viral Outbreaks Team

System: Linux   
Language: Python 3.8

Download the codes and install the requirements:

    mkdir $xxx
    cd $xxx

    git clone git@github.com:Mrzhouqifei/SUSTAIN.git
    git clone git@github.com:Mrzhouqifei/SUSTAIN_DATA.git
    pip install requirements.txt

open the runing files:

    cd SUSTAIN
    vim run.sh

modify "/home/zhouqifei" to your folder "$xxx"

    #！/bin/bash
    cd /home/zhouqifei/SUSTAIN
    today=$(date +%Y-%m-%d)
    echo "$today" >> log.txt 2>&1
    chmod +x update_data.py && /home/zhouqifei/anaconda3/bin/python update_data.py >> log.txt 2>&1
    chmod +x covid_forecast.py && /home/zhouqifei/anaconda3/bin/python covid_forecast.py >> log.txt 2>&1
    chmod +x restructed_data.py && /home/zhouqifei/anaconda3/bin/python restructed_data.py 
    cp -rf output/Roland/covid_forecast /home/zhouqifei/SUSTAIN_DATA
    cp -rf output/Roland/newest_policies /home/zhouqifei/SUSTAIN_DATA
    cp -rf output/Roland/covid_new_cases /home/zhouqifei/SUSTAIN_DATA

    cd /home/zhouqifei/SUSTAIN_DATA
    git pull
    git add .
    git commit -m "$today"
    git push

run the model and upload the SUSTAIN DATA to Github

    sh run.sh