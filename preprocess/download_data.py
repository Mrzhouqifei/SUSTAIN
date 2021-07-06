import time

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
    time.sleep(60)

    usa_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_' + series_category + '_US.csv'
    usa = pd.read_csv(usa_url)
    usa = usa.drop(
        ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State', 'Country_Region', 'Lat', 'Long_',
         'Combined_Key'], axis=1)
    df_usa = pd.DataFrame(usa.sum(axis=0)).T
    df_usa['Province/State'] = ''
    df_usa['Country/Region'] = 'United States'
    df_usa['Lat'] = 0
    df_usa['Long'] = 0

    df = pd.concat([df, df_usa])
    df.to_csv('raw_data/COVID/time_series_covid19_' + series_category + '_global_raw.csv', index=False)
    return df


def download_policy():
    url = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv"
    df = pd.read_csv(url)
    df.to_csv('raw_data/policies_latest.csv', index=False)
    return df


def policy_preprocess():
    policy_column = [
        'CountryName', 'CountryCode', 'Date',
        'C1_School closing',
        'C2_Workplace closing', 'C3_Cancel public events',
        'C4_Restrictions on gatherings', 'C5_Close public transport',
        'C6_Stay at home requirements',
        'C7_Restrictions on internal movement',
        'C8_International travel controls', 'E1_Income support',
        'E2_Debt/contract relief', 'E3_Fiscal measures',
        'E4_International support', 'H1_Public information campaigns',
        'H2_Testing policy', 'H3_Contact tracing',
        'H4_Emergency investment in healthcare', 'H5_Investment in vaccines',
        'H6_Facial Coverings', 'H7_Vaccination policy',
        'H8_Protection of elderly people',
        # 'StringencyIndex', 'ContainmentHealthIndex',
        # 'StringencyIndexForDisplay', 'StringencyLegacyIndex',
        # 'StringencyLegacyIndexForDisplay', 'GovernmentResponseIndex',
        # 'GovernmentResponseIndexForDisplay',
        # 'ContainmentHealthIndexForDisplay', 'EconomicSupportIndex',
        # 'EconomicSupportIndexForDisplay'
    ]
    df = pd.read_csv('raw_data/policies_latest.csv')
    df = df.dropna(axis=1, how='all')
    df = df[df.Jurisdiction == 'NAT_TOTAL']
    policy_column = [x for x in policy_column if x in df.columns]
    df = df[policy_column]
    df = df.rename(columns={'CountryName': 'entity', 'CountryCode': 'iso', 'Date': 'date'})
    df['date'] = pd.to_datetime(df.date.apply(str))
    # df = df.fillna(0)
    df = df.dropna()

    df.to_csv('raw_data/policies_all_countries_raw.csv', index=False)
    # indicators = pd.read_excel('raw_data/indicators.xlsx')
    # indicators = pd.read_excel('raw_data/SUSTAIN model indicator data, as of July 5, 2021_OVERALL_World.xlsx')
    # indicator_years = ['x2014', 'x2015', 'x2016', 'x2017', 'x2018', 'x2019', 'x2020', 'x2021']
    # indicators[indicator_years] = indicators[indicator_years].ffill(axis=1)
    # indicators = indicators[['category', 'indicator_clean_name', 'country', 'unit', 'x2021']].dropna()
    # indicators = indicators.rename(columns={'country': 'Country', 'indicator_clean_name': 'Indicator',
    #                                         'category': 'Category', 'unit': 'Unit'})
    #
    # country = pd.DataFrame(list(set(indicators.Country)), columns=['entity'])
    # df = df.merge(country, on='entity')
    # df.to_csv('raw_data/policies.csv', index=False)