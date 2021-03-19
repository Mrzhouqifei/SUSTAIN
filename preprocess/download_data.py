import pandas as pd
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download_policy_indicator(parent_folder_id='1QWz0egRt0d39MRdUT4OgkPaiptBHTOGN'):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    # 设置你在Bash上的下载路径。
    parent_folder_dir = './'
    if parent_folder_dir[-1] != '/':
        parent_folder_dir = parent_folder_dir + '/'

    file_dict = dict()
    folder_queue = [parent_folder_id]
    cnt = 0

    while len(folder_queue) != 0:
        current_folder_id = folder_queue.pop(0)
        file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(current_folder_id)}).GetList()

        for file1 in file_list:
            file_dict[cnt] = dict()
            file_dict[cnt]['id'] = file1['id']
            file_dict[cnt]['title'] = file1['title']

            cnt += 1
    for item in file_dict.values():
        download_file_from_google_drive(item['id'], 'raw_data/' + item['title'])


def download_covid_data(series_category):
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_' + series_category + '_global.csv'
    df = pd.read_csv(url)
    df.to_csv('raw_data/COVID/time_series_covid19_' + series_category + '_global.csv', index=False)
    return df


def download_policy():
    url = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv"
    df = pd.read_csv(url)
    df.to_csv('raw_data/policies.csv', index=False)
    return df
