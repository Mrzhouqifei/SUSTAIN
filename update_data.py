import pandas as pd

from preprocess.download_data import download_covid_data, download_policy, policy_preprocess
import time


def fix_countries_name():
    df = pd.read_csv('raw_data/COVID/time_series_covid19_confirmed_global_raw.csv')
    df['Country/Region'] = df['Country/Region'].str.replace('Taiwan\\*', 'Taiwan').str.replace(
        'Russia', 'Russian Federation').str.replace('Burma', 'Myanmar [Burma]').str.replace(
            'Korea, South', 'South Korea').str.replace('Vietnam', 'Viet Nam').str.replace('Iran', 'Iran (Islamic Republic of)').str.replace('Cabo Verde', 'Cape Verde').str.replace(
                        'Cote d\'Ivoire', 'Cote d\'Ivoire (Ivory Coast)').str.replace('Czechia', 'Czech Republic')
    # Myanmar [Burma] Viet Nam Slovakia Kyrgyzstan -> Kyrgyz Republic Iran -> Iran (Islamic Republic of) Cabo Verde -> Cape Verde
    # Cote d'Ivoire -> Cote d'Ivoire (Ivory Coast) Czechia -> Czech Republic

    df.loc[df['Province/State']=='Hong Kong','Country/Region'] = 'Hong Kong'
    df.to_csv('raw_data/COVID/time_series_covid19_confirmed_global.csv', index=False)

    df1 = pd.read_csv('raw_data/policies_all_countries_raw.csv')
    df1['entity'] = df1['entity'].str.replace('Russia', 'Russian Federation').str.replace(
        'Myanmar', 'Myanmar [Burma]').str.replace('Vietnam', 'Viet Nam').str.replace('Kyrgyz Republic', 'Kyrgyzstan').str.replace(
            'Iran', 'Iran (Islamic Republic of)').str.replace('Cote d\'Ivoire', 'Cote d\'Ivoire (Ivory Coast)').str.replace(
                'Slovak Republic', 'Slovakia')
    # df1 = df1.rename(columns={'Russia': 'Russian Federation'})
    df1.to_csv('raw_data/policies_all_countries.csv', index=False)


def covid_update():
    for series_category in ['confirmed']:   # ['recovered', 'confirmed', 'deaths']:
        retry = 3
        while retry > 0:
            try:
                download_covid_data(series_category)
                print('download success for %s' % series_category)
                break
            except:
                print('download fail for %s' % series_category)
            time.sleep(60)
            retry -= 1

    retry = 3
    while retry > 0:
        try:
            download_policy()
            print('download success for policy')
            # process the raw policies
            policy_preprocess()
            break
        except:
            print('download false for policy')
        time.sleep(60)
        retry -= 1
    fix_countries_name()



if __name__ == '__main__':
    # covid_update()  # update data
    # policy_preprocess()
    fix_countries_name()