from preprocess.download_data import download_covid_data, download_policy
import time

for series_category in ['confirmed', 'deaths', 'recovered']:
    try:
        download_covid_data(series_category)
        print('download success for %s' % series_category)
    except:
        print('download fail for %s' % series_category)
    time.sleep(5)

try:
    download_policy()
    print('download success for policy')
except:
    print('download false for policy')