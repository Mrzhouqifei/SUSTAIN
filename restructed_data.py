import pandas as pd
import numpy as np
import json
import datetime
import os


def mixed_policies_forecast(business_policy_intensity, mixed_policy_intensity,
                            newest_real_contagion, business_contagion_series):
    """
    intensity 1-5
    :param business_policy_intensity: dict, {'H7_Vaccination policy': 2, 'C5_Close public transport':3, ...}
    :param mixed_policy_intensity: dict, {'H7_Vaccination policy': 2, 'C5_Close public transport':3, ...}
    :param newest_real_contagion: float, the newest contagion rate now
    :param business_contagion_series: list, the "business as usual" contagion series of a specific country.
    :return: mixed_policies_contagion_series
    """
    policy_alphas = {   # intensity increase 1 unit, the contagion delta will multiply alpha
        'H7_Vaccination policy': 0.976,
        'C8_International travel controls': 0.979,
        'E1_Income support': 0.981,
        'H6_Facial Coverings': 0.982,
        'C5_Close public transport': 0.983,
        'H2_Testing policy': 0.984,
        'C4_Restrictions on gatherings': 0.985,
        'C2_Workplace closing': 0.986,
        'H3_Contact tracing': 0.988,
        'H8_Protection of elderly people': 0.989,
        'E2_Debt/contract relief': 0.991,
        'C7_Restrictions on internal movement': 0.991,
        'C6_Stay at home requirements': 0.993,
        'C3_Cancel public events': 0.995,
        'H1_Public information campaigns': 0.996,
        'C1_School closing': 0.998
    }

    delta_y = []
    pre_y = newest_real_contagion
    for y in business_contagion_series:
        delta_y.append(y - pre_y)
        pre_y = y

    policy_names = list(business_policy_intensity.keys())
    for policy in policy_names:
        alpha = policy_alphas[policy]
        intensity_change = mixed_policy_intensity[policy] - business_policy_intensity[policy]
        delta_cg = 1
        if intensity_change < 0:
            delta_cg = (1 / alpha) ** abs(intensity_change)
        elif intensity_change > 0:
            delta_cg = alpha ** intensity_change
        delta_y = [delta_cg * x for x in delta_y]

    mixed_policy_series = []
    pre_y = newest_real_contagion
    for item in delta_y:
        y = pre_y + item
        mixed_policy_series.append(y)
        pre_y = y
    return mixed_policy_series


def mixed_policies_newcases_forecast(business_policy_intensity, mixed_policy_intensity, business_newcases_series, population):
    """
    intensity 1-5
    :param business_policy_intensity: dict, {'H7_Vaccination policy': 2, 'C5_Close public transport':3, ...}
    :param mixed_policy_intensity: dict, {'H7_Vaccination policy': 2, 'C5_Close public transport':3, ...}
    :param newest_real_contagion: float, the newest contagion rate now
    :param business_contagion_series: list, the "business as usual" contagion series of a specific country.
    :return: mixed_policies_contagion_series
    """
    policy_alphas = {   # intensity increase 1 unit, the contagion delta will multiply alpha
        'H7_Vaccination policy': 0.976,
        'C8_International travel controls': 0.979,
        'E1_Income support': 0.981,
        'H6_Facial Coverings': 0.982,
        'C5_Close public transport': 0.983,
        'H2_Testing policy': 0.984,
        'C4_Restrictions on gatherings': 0.985,
        'C2_Workplace closing': 0.986,
        'H3_Contact tracing': 0.988,
        'H8_Protection of elderly people': 0.989,
        'E2_Debt/contract relief': 0.991,
        'C7_Restrictions on internal movement': 0.991,
        'C6_Stay at home requirements': 0.993,
        'C3_Cancel public events': 0.995,
        'H1_Public information campaigns': 0.996,
        'C1_School closing': 0.998
    }
    delta_y = [x/population for x in business_newcases_series]

    policy_names = list(business_policy_intensity.keys())
    for policy in policy_names:
        alpha = policy_alphas[policy]
        intensity_change = mixed_policy_intensity[policy] - business_policy_intensity[policy]
        delta_cg = 1
        if intensity_change < 0:
            delta_cg = (1 / alpha) ** abs(intensity_change)
        elif intensity_change > 0:
            delta_cg = alpha ** intensity_change
        delta_y = [delta_cg * x for x in delta_y]

    mixed_policy_series = [int(x*population) for x in delta_y]
    return mixed_policy_series


def re_construct_policy_rank_score():
    policy_rank_score = pd.read_excel('output/compiled_policy ranking and scores.xlsx').fillna('NULL')
    if not os.path.exists('output/Roland/policy_heat_map/countries'):
        os.makedirs('output/Roland/policy_heat_map/countries')

    labels = ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6', 'Month 7', 'Month 8', 'Month 9', 'Month 10', 'Month 11', 'Month 12', 'Month 13', 'Month 14', 'Month 15', 'Month 16', 'Month 17', 'Month 18', 'Overall']
    for key, group in policy_rank_score.groupby('policy'):
        res_dict = {}
        res_dict['source'] = "First confirmed case\\policy strictness score and ranking.dta"
        res_dict['build_id'] = str(datetime.date.today())
        res_dict["name"] = "DataSet"
        res_dict['data'] = []
        for key1 in labels:
            group1 = group[group['label']==key1]
            tmp_json = group1[['iso', 'score', 'rank']].to_dict(orient='records')
            single = {
                'label':key1,
                'data': tmp_json
            }
            res_dict['data'].append(single)
        res_json = json.dumps(res_dict,indent=1)
        f2 = open('output/Roland/policy_heat_map/' + key + '.json', 'w')
        f2.write(res_json)
        f2.close()
    policy_rank_score = policy_rank_score[policy_rank_score['label']=='Overall']
    for key, group in policy_rank_score.groupby('iso'):
        res_dict = {}
        res_dict['source'] = "First confirmed case\\policy strictness score and ranking.dta"
        res_dict['build_id'] = str(datetime.date.today())
        res_dict["name"] = "DataSet"
        res_dict['data'] = []
        for key1 in ['Overall']:
            group1 = group[group['label']==key1]
            tmp_json = group1[['policy', 'score', 'rank']].to_dict(orient='records')
            single = {
                'label':key1,
                'data': tmp_json
            }
            res_dict['data'].append(single)
        res_json = json.dumps(res_dict,indent=1)
        f2 = open('output/Roland/policy_heat_map/countries/' + key + '.json', 'w')
        f2.write(res_json)
        f2.close()


def re_construct_top_policy():
    policy_score_per_country=pd.read_csv('output/policy_effectiveness/policy_score_per_country.csv')
    policy_score_per_country=policy_score_per_country[(policy_score_per_country['Policy'] != 'H8_Protection of elderly people')
                                                      & (policy_score_per_country['Policy'] != 'E4_International support')
                                                      & (policy_score_per_country['Policy'] != 'H5_Investment in vaccines')
                                                      & (policy_score_per_country['Policy'] != 'E3_Fiscal measures')
                                                      & (policy_score_per_country['Policy'] != 'H4_Emergency investment in healthcare')]

    policy_score_per_country=population.merge(policy_score_per_country, on='Country').fillna('NULL')
    for key, group in policy_score_per_country.groupby('ISO'):
        if not os.path.exists('output/Roland/top_policy'):
            os.makedirs('output/Roland/top_policy')
        key = key.replace('/', '_')
        group = group.sort_values('score', ascending=False)
        group['rank'] = range(1, len(group) + 1)
        group[['ISO', 'Policy', 'score', 'rank']].to_csv('output/Roland/top_policy/' + key +'.csv', index=False)

        group = group[['ISO', 'Policy', 'rank']].to_dict(orient='records')
        res_json = json.dumps(group, indent=1)
        f2 = open('output/Roland/top_policy/' + key + '.json', 'w')
        f2.write(res_json)
        f2.close()


def re_construct_covid_forecast():
    history = pd.read_csv('output/covid_forecast/contagion/history.csv', index_col=0).unstack().reset_index().rename(
        columns={'level_0': 'Country', 'level_1': 'Date', 0: 'history'})
    lower = pd.read_csv('output/covid_forecast/contagion/prediction_lower.csv',
                        index_col=0).unstack().reset_index().rename(
        columns={'level_0': 'Country', 'level_1': 'Date', 0: 'lower'})
    average = pd.read_csv('output/covid_forecast/contagion/prediction_median.csv',
                          index_col=0).unstack().reset_index().rename(
        columns={'level_0': 'Country', 'level_1': 'Date', 0: 'average'})
    upper = pd.read_csv('output/covid_forecast/contagion/prediction_upper.csv',
                        index_col=0).unstack().reset_index().rename(
        columns={'level_0': 'Country', 'level_1': 'Date', 0: 'upper'})
    policies = pd.read_csv('raw_data/policies_all_countries.csv').drop(['iso', 'H8_Protection of elderly people',
                                                                            'E4_International support',
                                                                            'H5_Investment in vaccines',
                                                                            'E3_Fiscal measures',
                                                                            'H4_Emergency investment in healthcare'], axis=1).rename(
        columns={'entity': 'Country', 'date': 'Date'})
    population_year = 2020
    population = pd.read_excel('raw_data/UN Population Data, 1950 to 2020_Worldwide.xls', skiprows=[0, 1, 2])[
        ['country', 'iso', population_year]]
    population[population_year] = population[population_year] * 1000
    population = population.rename(columns={'country': 'Country', 'iso': 'ISO', 2020: 'Population'})

    data = policies.merge(history, on=['Country', 'Date'], how='right').ffill()
    data = data.merge(lower, on=['Country', 'Date'], how='outer')
    data = data.merge(average, on=['Country', 'Date'], how='outer')
    data = data.merge(upper, on=['Country', 'Date'], how='outer')
    data = data.merge(population, on=['Country'])

    data = data.fillna('NULL')
    data = data[['ISO', 'Date', 'history', 'lower', 'average', 'upper']]

    for key, group in data.groupby('ISO'):
        if not os.path.exists('output/Roland/covid_forecast'):
            os.makedirs('output/Roland/covid_forecast')
        # group.to_csv('output/Roland/covid_forecast/' + key + '.csv', index=False)
        group = group.to_dict(orient='records')
        res_json = json.dumps(group, indent=1)
        f2 = open('output/Roland/covid_forecast/' + key + '.json', 'w')
        f2.write(res_json)
        f2.close()

def re_construct_covid_new_cases():
    weakest_mixed_policy_intensity = {   # intensity increase 1 unit, the contagion delta will multiply alpha
        'H7_Vaccination policy': 0,
        'C8_International travel controls': 0,
        'E1_Income support': 0,
        'H6_Facial Coverings': 0,
        'C5_Close public transport': 0,
        'H2_Testing policy': 0,
        'C4_Restrictions on gatherings': 0,
        'C2_Workplace closing': 0,
        'H3_Contact tracing': 0,
        'E2_Debt/contract relief': 0,
        'C7_Restrictions on internal movement': 0,
        'C6_Stay at home requirements': 0,
        'C3_Cancel public events': 0,
        'H1_Public information campaigns': 0,
        'C1_School closing': 0,
    }

    strongest_mixed_policy_intensity = {   # intensity increase 1 unit, the contagion delta will multiply alpha
        'H7_Vaccination policy': 5,
        'C8_International travel controls': 4,
        'E1_Income support': 2,
        'H6_Facial Coverings': 4,
        'C5_Close public transport': 2,
        'H2_Testing policy': 3,
        'C4_Restrictions on gatherings': 4,
        'C2_Workplace closing': 3,
        'H3_Contact tracing': 2,
        'E2_Debt/contract relief': 2,
        'C7_Restrictions on internal movement': 2,
        'C6_Stay at home requirements': 3,
        'C3_Cancel public events': 2,
        'H1_Public information campaigns': 2,
        'C1_School closing': 3,
    }
    history = pd.read_csv('output/covid_forecast/contagion/history.csv', index_col=0).unstack().reset_index().rename(
        columns={'level_0': 'Country', 'level_1': 'Date', 0: 'history'})
    last_date = history['Date'].iloc[-1]
    average = pd.read_csv('output/covid_forecast/contagion/prediction_median.csv',
                        index_col=0).unstack().reset_index().rename(
        columns={'level_0': 'Country', 'level_1': 'Date', 0: 'average'})
    policies = pd.read_csv('raw_data/policies_all_countries.csv').drop(['iso', 'H8_Protection of elderly people',
                                                                            'E4_International support',
                                                                            'H5_Investment in vaccines',
                                                                            'E3_Fiscal measures',
                                                                            'H4_Emergency investment in healthcare'], axis=1).rename(
        columns={'entity': 'Country', 'date': 'Date'})
    population_year = 2020
    population = pd.read_excel('raw_data/UN Population Data, 1950 to 2020_Worldwide.xls', skiprows=[0, 1, 2])[
        ['country', 'iso', population_year]]
    population[population_year] = population[population_year] * 1000
    population = population.rename(columns={'country': 'Country', 'iso': 'ISO', 2020: 'Population'})

    data = policies.merge(history, on=['Country', 'Date'], how='right').ffill()
    data = data.merge(average, on=['Country', 'Date'], how='outer')
    data = data.merge(population, on=['Country'])

    for key, group in data.groupby('ISO'):
        if not os.path.exists('output/Roland/covid_new_cases'):
            os.makedirs('output/Roland/covid_new_cases')
        newest_v = group['history'][group['Date']==last_date].values[0]

        bussiness_policies = group[list(strongest_mixed_policy_intensity.keys())].ffill().iloc[-1].to_dict()
        strongest_policy = mixed_policies_forecast(bussiness_policies, strongest_mixed_policy_intensity, newest_v, group['average'][group['Date']>last_date])
        weakest_policy = mixed_policies_forecast(bussiness_policies, weakest_mixed_policy_intensity, newest_v, group['average'][group['Date']>last_date])
        strongest_policy = pd.DataFrame(strongest_policy, columns=['strongest_policy'])
        weakest_policy = pd.DataFrame(weakest_policy, columns=['weakest_policy'])
        strongest_policy['Date'] = list(group['Date'][group['Date']>last_date])
        weakest_policy['Date'] = list(group['Date'][group['Date']>last_date])
        
        group = group.merge(strongest_policy, on=['Date'], how='outer')
        group = group.merge(weakest_policy, on=['Date'], how='outer')
        
        group['average'][group['Date']==last_date] = newest_v
        group['average'] = group['average'].diff()
        
        group['strongest_policy'][group['Date']==last_date] = newest_v
        group['strongest_policy'] = group['strongest_policy'].diff()
        
        group['weakest_policy'][group['Date']==last_date] = newest_v
        group['weakest_policy'] = group['weakest_policy'].diff()

        group['history'] = group['history'].diff()

        group['rolling'] = (group['history'].fillna(0) + group['average'].fillna(0)).rolling(7).mean()
        
        group['history'] = (group['history'] * group['Population']).apply(lambda x: str(int(x)) if pd.notna(x) else x)
        group['average'] = (group['average'] * group['Population']).apply(lambda x: str(int(x)) if pd.notna(x) else x)
        group['rolling_7days'] = (group['rolling'] * group['Population']).apply(lambda x: str(int(x)) if pd.notna(x) else x)
        group['strongest_policy'] = (group['strongest_policy'] * group['Population']).apply(lambda x: str(int(x)) if pd.notna(x) else x)
        group['weakest_policy'] = (group['weakest_policy'] * group['Population']).apply(lambda x: str(int(x)) if pd.notna(x) else x)

        group = group[['Date', 'history', 'rolling_7days', 'average', 'strongest_policy', 'weakest_policy']].fillna('NULL').iloc[1:].rename(
            columns={'average': 'bussiness_as_usual'})
        
        group.to_csv('output/Roland/covid_new_cases/' + key + '.csv', index=False)
        group = group.to_dict(orient='records')
        res_json = json.dumps(group, indent=1)
        f2 = open('output/Roland/covid_new_cases/' + key + '.json', 'w')
        f2.write(res_json)
        f2.close()


def re_construct_newest_policies():
    policies = pd.read_csv('raw_data/policies_all_countries.csv').drop(['entity', 'H8_Protection of elderly people',
                                                                        'E4_International support',
                                                                        'H5_Investment in vaccines',
                                                                        'E3_Fiscal measures',
                                                                        'H4_Emergency investment in healthcare'], axis=1).rename(
        columns={'date': 'Date', 'iso': 'ISO'}).ffill()
    
    population_year = 2020
    population = pd.read_excel('raw_data/UN Population Data, 1950 to 2020_Worldwide.xls', skiprows=[0, 1, 2])[
        ['country', 'iso', population_year]]
    population[population_year] = population[population_year] * 1000
    population = population.rename(columns={'country': 'Country', 'iso': 'ISO', 2020: 'Population'})

    data = policies.fillna('NULL')
    data = data[data.Date==max(data.Date)].drop(['Date'], axis=1)
    data = data.merge(population, on=['ISO'])

    for key, group in data.groupby('ISO'):
        if not os.path.exists('output/Roland/newest_policies'):
            os.makedirs('output/Roland/newest_policies')
        # group.to_csv('output/Roland/covid_forecast/' + key + '.csv', index=False)
        group = group.to_dict(orient='records')
        res_json = json.dumps(group, indent=1)
        f2 = open('output/Roland/newest_policies/' + key + '.json', 'w')
        f2.write(res_json)
        f2.close()


def re_construct_top_indicator():
    policy_indicator = pd.read_csv('output/policy_effectiveness/top_indicators_per_policy_overall.csv')
    policy_indicator=policy_indicator[(policy_indicator['Policy'] != 'H8_Protection of elderly people')
                                                      & (policy_indicator['Policy'] != 'E4_International support')
                                                      & (policy_indicator['Policy'] != 'H5_Investment in vaccines')
                                                      & (policy_indicator['Policy'] != 'E3_Fiscal measures')
                                                      & (policy_indicator['Policy'] != 'H4_Emergency investment in healthcare')]

    indicator_years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021][::-1]
    indicators_category = pd.read_excel('raw_data/SUSTAIN model indicator data, as of July 5, 2021_OVERALL_World.xlsx')
    indicators_category['country'] = indicators_category['country'].str.replace('United States of America', 'United States')
    # 根据单位进行group by去重
    res = []
    for key, group in indicators_category.groupby(['indicator', 'country']):
        group = group.sort_values('unit', ascending=False)
        res.append(pd.DataFrame(group.iloc[0]).T)
    indicators_category = pd.concat(res)

    indicator_array = indicators_category[indicator_years].values
    years = [np.nan] * indicator_array.shape[0]
    for i, year in enumerate(indicator_years):
        flag = pd.isna(indicator_array[:, i])
        for j in range(indicator_array.shape[0]):
            if np.isnan(years[j]) and not flag[j]:
                years[j] = int(year)
    indicators_category['year'] = years

    indicator_years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
    indicators_category[indicator_years] = indicators_category[indicator_years].ffill(axis=1)
    indicators_category = indicators_category[['category', 'indicator', 'country', 'unit', 'year', 2021]]

    indicators_category = indicators_category.rename(columns={'country': 'Country', 'indicator': 'Indicator',
                                            'category': 'Category', 2021: 'value'})
    indicators_category = indicators_category.merge(population, on=['Country'])
    data = indicators_category.merge(policy_indicator, on=['Indicator']).dropna(axis=0, subset=['value'])
    data['unit'] = data['unit'].fillna('score')

    data = data.fillna('NULL')
    for key1, group1 in data.groupby('Policy'):
        key1 = key1.replace('/', '_')
        if not os.path.exists('output/Roland/top_indicator/' + key1):
            os.makedirs('output/Roland/top_indicator/' + key1)
        for key2, group2 in group1.groupby('ISO'):
            # group2[['rank', 'Category', 'Indicator', 'Policy', 'Country', 'ISO', 'value', 'unit', 'year']].sort_values('rank').to_csv('output/Roland/top_indicator/' + key1 + '/' + key2 + '.csv', index=False)
            # group2[['rank', 'Category', 'Indicator', 'Policy', 'Country', 'ISO', 'value', 'unit', 'year']].sort_values('rank').to_json('output/Roland/top_indicator/' + key1 + '/' + key2 + '.json', orient='records')
            group2 = group2[['rank', 'Category', 'Indicator', 'Policy', 'Country', 'ISO', 'value', 'unit', 'year']].sort_values('rank').to_dict(orient='records')
            res_json = json.dumps(group2, indent=1)
            f2 = open('output/Roland/top_indicator/' + key1 + '/' + key2 + '.json', 'w')
            f2.write(res_json)
            f2.close()

if __name__ == '__main__':
    population = pd.read_excel('raw_data/UN Population Data, 1950 to 2020_Worldwide.xls', skiprows=[0, 1, 2])[
        ['country', 'iso']]
    population = population.rename(columns={'country': 'Country', 'iso': 'ISO'})
    # re_construct_policy_rank_score()

    # re_construct_top_policy()
    # re_construct_top_indicator()
    
    re_construct_covid_forecast()
    re_construct_newest_policies()
    re_construct_covid_new_cases()
