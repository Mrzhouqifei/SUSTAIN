from preprocess.download_data import download_covid_data, download_policy, policy_preprocess
import time


def covid_update():
    for series_category in ['recovered', 'confirmed', 'deaths']:
        try:
            download_covid_data(series_category)
            print('download success for %s' % series_category)
        except:
            print('download fail for %s' % series_category)
        time.sleep(60)

    try:
        download_policy()
        print('download success for policy')
        # process the raw policies
        policy_preprocess()
    except:
        print('download false for policy')


if __name__ == '__main__':
    covid_update()  # update data
    # policy_preprocess()