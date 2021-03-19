from preprocess.download_data import download_covid_data, download_policy

for series_category in ['confirmed', 'deaths', 'recovered']:
    try:
        download_covid_data(series_category)
    except:
        print('download false for %s' % series_category)

# download_policy()