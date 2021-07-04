import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.simplefilter("ignore")
from sklearn.tree import DecisionTreeRegressor


def discrete_score(data, score_level=True):
    # nums = len(data) // 5
    # discrete_score = []
    # for i in range(4, -1, -1):
    #     num = i * 20 + 10
    #     if num == 90:
    #         score_level = 'very high'
    #     elif num == 70:
    #         score_level = 'high'
    #     elif num == 50:
    #         score_level = 'medium'
    #     elif num == 30:
    #         score_level = 'low'
    #     elif num == 10:
    #         score_level = 'very low'
    #     discrete_score.extend([score_level] * nums)
    # data['score_level'] = discrete_score[:len(data)]
    # data['score_level'] = ['very high'] * 3 + ['high'] * 3 + ['medium'] * 4 \
    #                       + ['low'] * 3 + ['very low'] * 3
    if score_level:
        data['score_level'] = ['very high'] * 4 + ['high'] * 4 + ['medium'] * 4 \
                              + ['low'] * 4 + ['very low'] * 4

    scores = np.sort(np.random.randn(len(data)))[::-1]
    scores = scores - np.min(scores)
    scores = scores / sum(scores)
    data['score'] = scores
    return data

def pre_series(series, countries):
    process_series = series.groupby('Country/Region').sum().drop(['Lat', 'Long'], axis=1).loc[countries, :].T[
        countries].fillna(0)
    process_series.index = pd.to_datetime(process_series.index)
    return process_series


def indicator_analysis(population, confirmed, indicators):
    countries = sorted(list((set(confirmed['Country/Region'])).intersection(
        set(indicators['Country'])).intersection(population.columns)))

    population = population[countries]
    confirmed = pre_series(confirmed, countries)
    # deaths = pre_series(deaths, countries)
    # recovered = pre_series(recovered, countries)

    # mortality/recovery rate 死亡率恢复率
    # mortality = (deaths / confirmed).replace([float('inf'), np.nan]).dropna(how='all')
    # recovery = (recovered / confirmed).replace([float('inf'), np.nan]).dropna(how='all')

    # contagion rate
    contagion = confirmed.values / population.values
    contagion = pd.DataFrame(contagion, columns=countries, index=confirmed.index)

    def indicator_influence(indicators, rates):
        """
        Perason correlation
        """
        indicators_values = indicators.pivot_table(index=['Indicator', 'Unit'], columns='Country', values=indicator_year)[
            countries].fillna(0).T
        indicator_mortality = {}
        for i in range(len(indicators_values.columns)):
            indicator_mortality[indicators_values.columns[i]] = (indicators_values.iloc[:, i].corr(rates.iloc[-1]))
        indicator_mortality = sorted(indicator_mortality.items(), key=lambda x: -abs(x[1]))
        rate = []
        for x in indicator_mortality:
            rate.append([x[0][0], x[0][1], x[1]])
        return pd.DataFrame(rate, columns=['Indicator', 'Unit', 'Correlation'])

    # indicator_influence(indicators, mortality).to_csv('output/policy_effectiveness/indicator_influence/indicator_mortality.csv',
    #                                                   index=False)
    # indicator_influence(indicators, recovery).to_csv('output/policy_effectiveness/indicator_influence/indicator_recovery.csv',
    #                                                  index=False)
    indicator_influence(indicators, contagion).to_csv('output/policy_effectiveness/indicator_coefficient_global.csv',
                                                     index=False)


def policy_analysis(population, confirmed, policies):
    countries = sorted(list((set(confirmed['Country/Region'])).intersection(set(policies.entity)).intersection(population.columns)))

    population = population[countries]
    confirmed = pre_series(confirmed, countries)
    # deaths = pre_series(deaths, countries)
    # recovered = pre_series(recovered, countries)

    # mortality/recovery rate 死亡率恢复率
    # mortality = (deaths / confirmed).replace([float('inf'), np.nan]).dropna(how='all').stack().reset_index()
    # recovery = (recovered / confirmed).replace([float('inf'), np.nan]).dropna(how='all').stack().reset_index()
    # mortality.columns = ['date', 'country', 'y']
    # recovery.columns = ['date', 'country', 'y']

    # contagion rate
    contagion = confirmed.values / population.values
    contagion = pd.DataFrame(contagion, columns=countries, index=confirmed.index).stack().reset_index()
    contagion.columns = ['date', 'country', 'y']
    # contagion['y'] = contagion['y'].diff()

    def policy_influence(policies, rates):
        policy_names = [x for x in list(policies.columns) if x not in ['entity', 'iso', 'date']]
        for policy_name in policy_names:
            """
            information gain based on Gini Index
            """
            tmp = policies[['entity', 'date', policy_name]].pivot_table(index=['date', 'entity'],
                                                                        values=policy_name).reset_index()
            tmp.columns = ['date', 'country', policy_name]
            tmp.date = pd.to_datetime(tmp.date)
            rates = rates.merge(tmp, on=['date', 'country'], how='inner').dropna()

        model = DecisionTreeRegressor().fit(rates[policy_names], rates['y'])
        model_score = model.tree_.compute_feature_importances(normalize=True)
        res = pd.DataFrame({'Policy': policy_names, 'score': model_score}).sort_values('score', ascending=False)
        res = discrete_score(res)
        return res

    # policy_influence(policies, mortality).to_csv('output/policy_effectiveness/policy_influence/policy_mortality.csv', index=False)
    # policy_influence(policies, recovery).to_csv('output/policy_effectiveness/policy_influence/policy_recovery.csv', index=False)
    policy_influence(policies, contagion).to_csv('output/policy_effectiveness/policy_score_global.csv', index=False)


def policy_country_analysis(population, confirmed, policies):  # deaths, recovered,
    countries = sorted(list((set(confirmed['Country/Region'])).intersection(set(policies.entity)).intersection(population.columns)))
    print(countries)

    population = population[countries]
    confirmed = pre_series(confirmed, countries)
    # deaths = pre_series(deaths, countries)
    # recovered = pre_series(recovered, countries)

    # mortality/recovery rate 死亡率恢复率
    # mortality = (deaths / confirmed).replace([float('inf'), np.nan]).dropna(how='all').stack().reset_index()
    # recovery = (recovered / confirmed).replace([float('inf'), np.nan]).dropna(how='all').stack().reset_index()
    # mortality.columns = ['date', 'country', 'y']
    # recovery.columns = ['date', 'country', 'y']

    # contagion rate
    contagion = confirmed.values / population.values
    contagion = pd.DataFrame(contagion, columns=countries, index=confirmed.index).stack().reset_index()
    contagion.columns = ['date', 'country', 'y']
    # contagion['y'] = contagion['y'].diff()

    def policy_influence(policies, rates):
        policy_names = [x for x in list(policies.columns) if x not in ['entity', 'iso', 'date']]
        for policy_name in policy_names:
            """
            information gain based on Gini Index
            """
            tmp = policies[['entity', 'date', policy_name]].pivot_table(index=['date', 'entity'],
                                                                        values=policy_name).reset_index()
            tmp.columns = ['date', 'country', policy_name]
            tmp.date = pd.to_datetime(tmp.date)
            rates = rates.merge(tmp, on=['date', 'country'], how='inner').dropna()
        res = None
        for key, rate in rates.groupby('country'):
            if len(rate) > 50:
                model = DecisionTreeRegressor().fit(rate[policy_names], rate['y'])
                model_score = model.tree_.compute_feature_importances(normalize=True)
                score = pd.DataFrame({'Policy': policy_names, 'score': model_score}).sort_values(
                    'score', ascending=False)
                score['Country'] = key
                score = discrete_score(score)
                res = pd.concat((res, score[['Country', 'Policy', 'score', 'score_level']]))  # [:10]
        return res

    # policy_influence(policies, mortality).to_csv('output/policy_effectiveness/country_policy_influence/policy_mortality.csv',
    #                                              index=False)
    # policy_influence(policies, recovery).to_csv('output/policy_effectiveness/country_policy_influence/policy_recovery.csv',
    #                                             index=False)
    policy_influence(policies, contagion).to_csv('output/policy_effectiveness/policy_score_per_country.csv',
                                                index=False)


def policy_indicator_analysis(population, confirmed, policies, indicators):
    leading_indicators = [
    ]
    top_indicators = [
        'Employment in services (% of total employment) (modeled ILO estimate)',
        'Individuals using the Internet',
        'Interest rate on loans and discounts - public',
        'Incidence of tuberculosis (per 100,000 people)',
        'Press Freedom Rank',
        'Rural population',
        'Human Development Index',
        'Production index: Manufacturing',
        'investmentfreedom',
        'Net migration',
        'Fixed broadband subscriptions',
        'Labor force participation rate, male (% of male population ages 15+) (national estimate)',
        'Population ages 15-64 (% of total population)',
        'International tourism, receipts',
        'Mobile cellular subscriptions',
        'Population aged 65+ years (% of total population)',
        'Doctors per 1 000 inhabitants.',
        'Output per worker (GDP constant 2011 international $ in PPP -- ILO modelled estimates, Nov. 2019',
        'Foreign direct investment, net',
        'Political stability and absence of violence/terrorism estimate',
        'Labour force participation rate (Total) - Youth, adults: 15-64',
        'Expenditure on health (% of GDP)',
        'Educational attainment, at least Bachelor\'s or equivalent, population 25+, total (%) (cumulative)',
        'Proportion of seats held by women in national parliaments (%)',
        'Gini index (World Bank estimate)',
        'Proportion of Unemployment in Non-agricultural Employment (total)',
        'Labor force participation rate, female (% of female population ages 15+) (modeled ILO estimate)',
        'Volume of exports',
        'Volume of imports',
        'Output per worker (GDP constant 2011 international $ in PPP -- ILO modelled estimates, Nov. 2019',
        'Interest rate on loans and discounts',
        'Unemployment, total (% of total labor force) (modeled ILO estimate)',
        'CPIA transparency, accountability, and corruption in the public sector rating (1=low to 6=high)',
        'Age dependency ratio (% of working-age population)',
        'CO2 emissions (metric tons per capita)',
        'Starting a Business (Score)',
        'CPI score',
    ]
    leading_indicators = pd.DataFrame(list(set(leading_indicators + top_indicators)), columns=['Indicator'])
    # indicators = indicators[indicators.Indicator in leading_indicators]
    indicators = indicators.merge(leading_indicators, on=['Indicator'])
    # 根据单位进行group by去重
    res = []
    for key, group in indicators.groupby(['Indicator', 'Country']):
        group = group.sort_values('Unit')
        res.append(pd.DataFrame(group.iloc[0]).T)
    indicators = pd.concat(res).fillna(0)

    countries = sorted(list(
        (set(confirmed['Country/Region'])).intersection(set(policies.entity)).intersection(
            set(indicators['Country'])).intersection(population.columns)))

    population = population[countries]
    confirmed = pre_series(confirmed, countries)
    # deaths = pre_series(deaths, countries)
    # recovered = pre_series(recovered, countries)

    # mortality/recovery rate 死亡率恢复率
    # mortality = (deaths / confirmed).replace([float('inf'), np.nan]).dropna(how='all').stack().reset_index()
    # recovery = (recovered / confirmed).replace([float('inf'), np.nan]).dropna(how='all').stack().reset_index()
    # mortality.columns = ['date', 'country', 'y']
    # recovery.columns = ['date', 'country', 'y']

    # contagion rate
    contagion = confirmed.values / population.values
    contagion = pd.DataFrame(contagion, columns=countries, index=confirmed.index).stack().reset_index()
    contagion.columns = ['date', 'country', 'y']

    def policy_indicators(policies, rates):
        policy_names = [x for x in list(policies.columns) if x not in ['entity', 'iso', 'date']]
        for policy_name in policy_names:
            """
            information gain based on Gini Index
            """
            tmp = policies[['entity', 'date', policy_name]].pivot_table(index=['date', 'entity'],
                                                                        values=policy_name).reset_index()
            tmp.columns = ['date', 'country', policy_name]
            tmp.date = pd.to_datetime(tmp.date)
            rates = rates.merge(tmp, on=['date', 'country'], how='inner').dropna()
        indicators_values = \
        indicators.pivot_table(index=['Indicator', 'Unit'], columns='Country', values=indicator_year)[countries].fillna(
            0).T

        res = None
        for policy_name in policy_names:
            train = indicators_values.merge(rates[['date', 'country', 'y', policy_name]], left_index=True,
                                            right_on='country')
            train_columns = [x for x in train.columns if x not in ['date', 'country', 'y']]
            model = DecisionTreeRegressor().fit(train[train_columns], train['y'])
            model_score = model.tree_.compute_feature_importances(normalize=True)
            score = pd.DataFrame({'Indicator': train_columns, 'score': model_score}).sort_values(
                'score', ascending=False)
            score = score[score['Indicator'] != policy_name][:20]
            score['Policy'] = policy_name
            score['rank'] = list(range(1, len(score) + 1))
            # score = discrete_score(score)
            res = pd.concat((res, score[['Policy', 'Indicator', 'score', 'rank']]))  # 抽取几个因子
        return res

    # policy_indicators(policies, contagion).to_csv('output/policy_effectiveness/top_indicators_per_policy.csv', index=False)
    return policy_indicators(policies, contagion)


def policy_indicator_analysis_all(population, confirmed, policies, indicators):
    top_indicators = policy_indicator_analysis(population.copy(), confirmed.copy(),
                                               policies.copy(), indicators.copy()).drop(['score'], axis=1)

    # 根据单位进行group by去重
    res = []
    for key, group in indicators.groupby(['Indicator', 'Country']):
        group = group.sort_values('Unit')
        res.append(pd.DataFrame(group.iloc[0]).T)
    indicators = pd.concat(res).fillna(0)

    countries = sorted(list(
        (set(confirmed['Country/Region'])).intersection(set(policies.entity)).intersection(
            set(indicators['Country'])).intersection(population.columns)))

    population = population[countries]
    confirmed = pre_series(confirmed, countries)

    # contagion rate
    contagion = confirmed.values / population.values
    contagion = pd.DataFrame(contagion, columns=countries, index=confirmed.index).stack().reset_index()
    contagion.columns = ['date', 'country', 'y']

    def policy_indicators(policies, rates):
        policy_names = [x for x in list(policies.columns) if x not in ['entity', 'iso', 'date']]
        for policy_name in policy_names:
            """
            information gain based on Gini Index
            """
            tmp = policies[['entity', 'date', policy_name]].pivot_table(index=['date', 'entity'],
                                                                        values=policy_name).reset_index()
            tmp.columns = ['date', 'country', policy_name]
            tmp.date = pd.to_datetime(tmp.date)
            rates = rates.merge(tmp, on=['date', 'country'], how='inner').dropna()
        indicators_values = \
        indicators.pivot_table(index=['Indicator', 'Unit'], columns='Country', values=indicator_year)[countries].fillna(
            0).T

        res = None
        for policy_name in policy_names:
            train = indicators_values.merge(rates[['date', 'country', 'y', policy_name]], left_index=True,
                                            right_on='country')
            train_columns = [x for x in train.columns if x not in ['date', 'country', 'y']]
            model = DecisionTreeRegressor().fit(train[train_columns], train['y'])
            model_score = model.tree_.compute_feature_importances(normalize=True)
            score = pd.DataFrame({'Indicator': train_columns, 'score': model_score}).sort_values(
                'score', ascending=False)
            score = score[score['Indicator'] != policy_name][:1000]
            score['Policy'] = policy_name
            score['rank'] = list(range(1, len(score) + 1))
            # score = discrete_score(score)
            res = pd.concat((res, score[['Policy', 'Indicator', 'score', 'rank']]))  # 抽取几个因子
        return res

    all_indicators = policy_indicators(policies, contagion).drop(['score'], axis=1)
    all_indicators = all_indicators.rename(columns={'rank': 'rank_all'})
    top_indicators = top_indicators.rename(columns={'rank': 'rank_top'})
    new_indicators = top_indicators.merge(all_indicators, how='right', on=['Policy', 'Indicator'])
    res = []
    for key, group in new_indicators.groupby('Policy'):
        group = group.sort_values(['Policy', 'rank_top', 'rank_all'])
        group['rank'] = list(range(1, len(group) + 1))
        scores = np.sort(np.random.randn(len(group)))[::-1]
        scores = scores - np.min(scores)
        scores = scores / sum(scores)
        group['score'] = scores
        res.append(group)
    res = pd.concat(res)[['Policy', 'Indicator', 'score', 'rank']]
    res.to_csv('output/policy_effectiveness/top_indicators_per_policy.csv', index=False)

    # policy_indicators(policies, contagion).to_csv('output/policy_effectiveness/all_indicators_per_policy.csv', index=False)


def policy_indicator_analysis_overall(population, confirmed, policies, indicators):
    # 根据单位进行group by去重
    res = []
    for key, group in indicators.groupby(['Indicator', 'Country']):
        group = group.sort_values('Unit')
        res.append(pd.DataFrame(group.iloc[0]).T)
    indicators = pd.concat(res).fillna(0)

    countries = sorted(list(
        (set(confirmed['Country/Region'])).intersection(set(policies.entity)).intersection(
            set(indicators['Country'])).intersection(population.columns)))

    population = population[countries]
    confirmed = pre_series(confirmed, countries)

    # contagion rate
    contagion = confirmed.values / population.values
    contagion = pd.DataFrame(contagion, columns=countries, index=confirmed.index).stack().reset_index()
    contagion.columns = ['date', 'country', 'y']

    def policy_indicators(policies, rates):
        policy_names = [x for x in list(policies.columns) if x not in ['entity', 'iso', 'date']]
        for policy_name in policy_names:
            """
            information gain based on Gini Index
            """
            tmp = policies[['entity', 'date', policy_name]].pivot_table(index=['date', 'entity'],
                                                                        values=policy_name).reset_index()
            tmp.columns = ['date', 'country', policy_name]
            tmp.date = pd.to_datetime(tmp.date)
            rates = rates.merge(tmp, on=['date', 'country'], how='inner').dropna()
        indicators_values = \
        indicators.pivot_table(index=['Category', 'Indicator', 'Unit'], columns='Country', values=indicator_year)[countries].fillna(
            0).T
        res = None
        for policy_name in policy_names:
            train = indicators_values.merge(rates[['date', 'country', 'y', policy_name]], left_index=True,
                                            right_on='country')
            train_columns = [x for x in train.columns if x not in ['date', 'country', 'y']]
            model = DecisionTreeRegressor().fit(train[train_columns], train['y'])
            model_score = model.tree_.compute_feature_importances(normalize=True)
            score = pd.DataFrame({'Category_Indicator_Unit': train_columns,
                                  'score': model_score}).sort_values('score', ascending=False)
            score = score[score['Category_Indicator_Unit'] != policy_name]
            score['Policy'] = policy_name
            score['rank'] = list(range(1, len(score) + 1))
            score = discrete_score(score, False)
            res = pd.concat((res, score[['Policy', 'Category_Indicator_Unit', 'score', 'rank']]))  # 抽取几个因子
        return res
    # 字符处理
    res = policy_indicators(policies, contagion)
    res['Category_Indicator_Unit'] = res['Category_Indicator_Unit'].astype(dtype=str)
    res[['Category', 'Indicator', 'Unit']] = res['Category_Indicator_Unit'].str.split('\',', expand=True)
    res['Category'] = res['Category'].str.replace("\(\'", "")
    res['Unit'] = res['Unit'].str.replace("\'\)", "").str.replace(" \'", "").str.replace("\'", "")
    res['Indicator'] = res['Indicator'].str.replace(" \'", "").str.replace("\'", "")
    res[['Policy', 'Category', 'Indicator', 'Unit', 'rank', 'score']].to_excel('output/policy_effectiveness/'
                                                                             'top_indicators_per_policy_overall.xls', index=False)

def policy_indicator_analysis_category(population, confirmed, policies, indicators):
    # 根据单位进行group by去重
    res = []
    for key, group in indicators.groupby(['Indicator', 'Country']):
        group = group.sort_values('Unit')
        res.append(pd.DataFrame(group.iloc[0]).T)
    indicators = pd.concat(res).fillna(0)

    countries = sorted(list(
        (set(confirmed['Country/Region'])).intersection(set(policies.entity)).intersection(
            set(indicators['Country'])).intersection(population.columns)))

    population = population[countries]
    confirmed = pre_series(confirmed, countries)

    # contagion rate
    contagion = confirmed.values / population.values
    contagion = pd.DataFrame(contagion, columns=countries, index=confirmed.index).stack().reset_index()
    contagion.columns = ['date', 'country', 'y']

    def policy_indicators(policies, rates, indicators):
        policy_names = [x for x in list(policies.columns) if x not in ['entity', 'iso', 'date']]
        for policy_name in policy_names:
            """
            information gain based on Gini Index
            """
            tmp = policies[['entity', 'date', policy_name]].pivot_table(index=['date', 'entity'],
                                                                        values=policy_name).reset_index()
            tmp.columns = ['date', 'country', policy_name]
            tmp.date = pd.to_datetime(tmp.date)
            rates = rates.merge(tmp, on=['date', 'country'], how='inner').dropna()
        indicators_values = \
        indicators.pivot_table(index=['Category', 'Indicator', 'Unit'], columns='Country', values=indicator_year).fillna(
            0).T
        res = None
        for policy_name in policy_names:
            train = indicators_values.merge(rates[['date', 'country', 'y', policy_name]], left_index=True,
                                            right_on='country')
            train_columns = [x for x in train.columns if x not in ['date', 'country', 'y']]
            model = DecisionTreeRegressor().fit(train[train_columns], train['y'])
            model_score = model.tree_.compute_feature_importances(normalize=True)
            score = pd.DataFrame({'Category_Indicator_Unit': train_columns,
                                  'score': model_score}).sort_values('score', ascending=False)
            score = score[score['Category_Indicator_Unit'] != policy_name]
            score['Policy'] = policy_name
            score['rank'] = list(range(1, len(score) + 1))
            score = discrete_score(score, False)
            res = pd.concat((res, score[['Policy', 'Category_Indicator_Unit', 'score', 'rank']]))  # 抽取几个因子
        return res
    # 字符处理
    all_res = []
    for key, group in indicators.groupby('Category'):
        res = policy_indicators(policies, contagion, group)
        res['Category_Indicator_Unit'] = res['Category_Indicator_Unit'].astype(dtype=str)
        res[['Category', 'Indicator', 'Unit']] = res['Category_Indicator_Unit'].str.split('\',', expand=True)
        res['Category'] = res['Category'].str.replace("\(\'", "")
        res['Unit'] = res['Unit'].str.replace("\'\)", "").str.replace(" \'", "").str.replace("\'", "")
        res['Indicator'] = res['Indicator'].str.replace(" \'", "").str.replace("\'", "")
        all_res.append(res[['Policy', 'Category', 'Indicator', 'Unit', 'rank', 'score']])
    all_res = pd.concat(all_res)
    all_res.to_excel('output/policy_effectiveness/top_indicators_per_policy_category.xls', index=False)

if __name__ == '__main__':
    init_seed = 2020
    np.random.seed(init_seed)

    population_year = 2020

    # population = pd.read_excel('raw_data/PopulationData.xls', skiprows=[0, 1, 2])[['Country', population_year]]
    # population[population_year] = population[population_year] * 1000
    # population = population.set_index('Country').T
    # indicator_year = 'x2021'
    # indicator_years = ['x2014', 'x2015', 'x2016', 'x2017', 'x2018', 'x2019', 'x2020', 'x2021']
    # indicators = pd.read_excel('raw_data/SUSTAIN model indicator data, as of June 27, 2021_OVERALL.xlsx')
    # indicators[indicator_years] = indicators[indicator_years].ffill(axis=1)
    # indicators = indicators[['category', 'indicator_clean_name', 'country', 'unit', 'x2021']].dropna()
    # indicators = indicators.rename(columns={'country': 'Country', 'indicator_clean_name': 'Indicator',
    #                                         'category': 'Category', 'unit': 'Unit'})
    # indicators_category = pd.read_excel('raw_data/SUSTAIN model indicator data, as of June 27, 2021_CATEGORY.xlsx')
    # indicators_category[indicator_years] = indicators_category[indicator_years].ffill(axis=1)
    # indicators_category = indicators_category[['category', 'indicator_clean_name', 'country', 'unit', 'x2021']].dropna()
    # indicators_category = indicators_category.rename(columns={'country': 'Country', 'indicator_clean_name': 'Indicator',
    #                                         'category': 'Category', 'unit': 'Unit'})
    # policies = pd.read_csv('raw_data/policies.csv')

    population = pd.read_excel('raw_data/UN Population Data, 1950 to 2020_Worldwide.xls', skiprows=[0, 1, 2])[['country', population_year]]
    population[population_year] = population[population_year] * 1000
    population = population.rename(columns={'country': 'Country'})
    population = population.set_index('Country').T
    indicator_year = 2021
    indicator_years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
    indicators = pd.read_excel('raw_data/SUSTAIN model indicator data, as of July 3, 2021_OVERALL_World.xlsx')
    indicators[indicator_years] = indicators[indicator_years].ffill(axis=1)
    indicators = indicators[['category', 'indicator', 'country', 'unit', 2021]].dropna()
    indicators = indicators.rename(columns={'country': 'Country', 'indicator': 'Indicator',
                                            'category': 'Category', 'unit': 'Unit'})
    indicators_category = pd.read_excel('raw_data/SUSTAIN model indicator data, as of July 3, 2021_CATEGORY_World.xlsx')
    indicators_category[indicator_years] = indicators_category[indicator_years].ffill(axis=1)
    indicators_category = indicators_category[['category', 'indicator', 'country', 'unit', 2021]].dropna()
    indicators_category = indicators_category.rename(columns={'country': 'Country', 'indicator': 'Indicator',
                                            'category': 'Category', 'unit': 'Unit'})
    policies = pd.read_csv('raw_data/policies_all_countries.csv')

    # 'confirmed', 'deaths', 'recovered'
    confirmed = pd.read_csv('raw_data/COVID/time_series_covid19_' + 'confirmed' + '_global.csv')
    # deaths = pd.read_csv('raw_data/COVID/time_series_covid19_' + 'deaths' + '_global.csv')
    # recovered = pd.read_csv('raw_data/COVID/time_series_covid19_' + 'recovered' + '_global.csv')

    # indicator_analysis(population, confirmed, indicators)
    # policy_analysis(population, confirmed, policies)
    # policy_country_analysis(population, confirmed, policies)

    policy_indicator_analysis_overall(population, confirmed, policies, indicators)
    policy_indicator_analysis_category(population, confirmed, policies, indicators_category)