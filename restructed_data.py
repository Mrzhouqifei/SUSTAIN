import pandas as pd
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


def re_construct_heap_map():
    policy_score_per_country=pd.read_csv('output/policy_effectiveness/policy_score_per_country.csv')
    policy_score_per_country=population.merge(policy_score_per_country, on='Country')
    for key, group in policy_score_per_country.groupby('Policy'):
        if not os.path.exists('output/Roland/policy_heap_map'):
            os.makedirs('output/Roland/policy_heap_map')
        key = key.replace('/', '_')
        group.to_csv('output/Roland/policy_heap_map/' + key +'.csv', index=False)
        group[['ISO','score','score_level']].to_json('output/Roland/policy_heap_map/' + key + '.json', orient='records')


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
    policies = pd.read_csv('raw_data/policies_all_countries_raw.csv').drop('iso', axis=1).rename(
        columns={'entity': 'Country', 'date': 'Date'})
    population_year = 2020
    population = pd.read_excel('raw_data/UN Population Data, 1950 to 2020_Worldwide.xls', skiprows=[0, 1, 2])[
        ['country', population_year]]
    population[population_year] = population[population_year] * 1000
    population = population.rename(columns={'country': 'Country', 2020: 'Population'})

    data = policies.merge(history, on=['Country', 'Date'], how='right').ffill()
    data = data.merge(lower, on=['Country', 'Date'], how='outer')
    data = data.merge(average, on=['Country', 'Date'], how='outer')
    data = data.merge(upper, on=['Country', 'Date'], how='outer')
    data = data.merge(population, on=['Country'])

    for key, group in data.groupby('Country'):
        if not os.path.exists('output/Roland/covid_forecast'):
            os.makedirs('output/Roland/covid_forecast')
        group.to_csv('output/Roland/covid_forecast/' + key + '.csv', index=False)


def re_construct_top_indicator():
    policy_indicator = pd.read_csv('output/policy_effectiveness/top_indicators_per_policy_overall.csv')
    indicator_years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
    indicators_category = pd.read_excel('raw_data/SUSTAIN model indicator data, as of July 5, 2021_OVERALL_World.xlsx')
    indicators_category[indicator_years] = indicators_category[indicator_years].ffill(axis=1)
    indicators_category = indicators_category[['category', 'indicator', 'country', 'unit', 2021]].dropna()
    indicators_category = indicators_category.rename(columns={'country': 'Country', 'indicator': 'Indicator',
                                            'category': 'Category', 'unit': 'Unit', 2021:'value'})
    indicators_category = indicators_category.merge(population, on=['Country'])
    data = indicators_category.merge(policy_indicator, on=['Indicator'])

    for key1, group1 in data.groupby('Policy'):
        key1 = key1.replace('/', '_')
        if not os.path.exists('output/Roland/top_indicator/' + key1):
            os.makedirs('output/Roland/top_indicator/' + key1)
        for key2, group2 in group1.groupby('ISO'):
            group2[['Category','Indicator','Policy','Country','ISO','rank','value']].to_csv('output/Roland/top_indicator/' + key1 + '/' + key2 + '.csv', index=False)
            group2[['Category', 'Indicator', 'Policy', 'Country', 'ISO', 'rank', 'value']].to_json('output/Roland/top_indicator/' + key1 + '/' + key2 + '.json', orient='records')
            # for key3, group3 in group2.groupby('Category'):
            #     group3.to_csv('output/Roland/top_indicator/' + key1 + '/' + key2 + '/' + key3 + '.csv', index=False)


if __name__ == '__main__':
    population = pd.read_excel('raw_data/UN Population Data, 1950 to 2020_Worldwide.xls', skiprows=[0, 1, 2])[
        ['country', 'iso']]
    population = population.rename(columns={'country': 'Country', 'iso': 'ISO'})

    re_construct_heap_map()
    # re_construct_covid_forecast()
    re_construct_top_indicator()