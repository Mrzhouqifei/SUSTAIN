import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
from sklearn.tree import DecisionTreeRegressor


def pre_series(series, countries):
    process_series = series.groupby('Country/Region').sum().drop(['Lat', 'Long'], axis=1).loc[countries, :].T[
        countries].fillna(0)
    process_series.index = pd.to_datetime(process_series.index)
    return process_series


def indicator_analysis(confirmed, deaths, recovered, indicators):
    countries = sorted(list((set(confirmed['Country/Region'])).intersection(set(indicators['Country']))))
    confirmed = pre_series(confirmed, countries)
    deaths = pre_series(deaths, countries)
    recovered = pre_series(recovered, countries)

    # mortality/recovery rate 死亡率恢复率
    mortality = (deaths / confirmed).dropna(how='all')
    recovery = (recovered / confirmed).dropna(how='all')

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
        return pd.DataFrame(rate, columns=['Indicator', 'Unit', 'influence'])

    indicator_influence(indicators, mortality).to_csv('output/step_two/indicator_influence/indicator_mortality.csv',
                                                      index=False)
    indicator_influence(indicators, recovery).to_csv('output/step_two/indicator_influence/indicator_recovery.csv',
                                                     index=False)


def policy_analysis(confirmed, deaths, recovered, policies):
    countries = sorted(list((set(confirmed['Country/Region'])).intersection(set(policies.entity))))
    confirmed = pre_series(confirmed, countries)
    deaths = pre_series(deaths, countries)
    recovered = pre_series(recovered, countries)

    # mortality/recovery rate 死亡率恢复率
    mortality = (deaths / confirmed).dropna(how='all').stack().reset_index()
    recovery = (recovered / confirmed).dropna(how='all').stack().reset_index()
    mortality.columns = ['date', 'country', 'y']
    recovery.columns = ['date', 'country', 'y']

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
        model_importance = model.tree_.compute_feature_importances(normalize=True)
        return pd.DataFrame({'Policy': policy_names, 'importance': model_importance}).sort_values('importance',
                                                                                                  ascending=False)

    policy_influence(policies, mortality).to_csv('output/step_two/policy_influence/policy_mortality.csv', index=False)
    policy_influence(policies, recovery).to_csv('output/step_two/policy_influence/policy_recovery.csv', index=False)


def policy_country_analysis(confirmed, deaths, recovered, policies):
    countries = sorted(list((set(confirmed['Country/Region'])).intersection(set(policies.entity))))
    confirmed = pre_series(confirmed, countries)
    deaths = pre_series(deaths, countries)
    recovered = pre_series(recovered, countries)

    # mortality/recovery rate 死亡率恢复率
    mortality = (deaths / confirmed).dropna(how='all').stack().reset_index()
    recovery = (recovered / confirmed).dropna(how='all').stack().reset_index()
    mortality.columns = ['date', 'country', 'y']
    recovery.columns = ['date', 'country', 'y']

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
                model_importance = model.tree_.compute_feature_importances(normalize=True)
                importance = pd.DataFrame({'Policy': policy_names, 'importance': model_importance}).sort_values(
                    'importance', ascending=False)
                importance['Country'] = key
                res = pd.concat((res, importance[['Country', 'Policy', 'importance']])) # [:10]
        return res

    policy_influence(policies, mortality).to_csv('output/step_two/country_policy_influence/policy_mortality.csv',
                                                 index=False)
    policy_influence(policies, recovery).to_csv('output/step_two/country_policy_influence/policy_recovery.csv',
                                                index=False)


def policy_indicator_analysis(confirmed, deaths, recovered, policies, indicators):
    countries = sorted(list(
        (set(confirmed['Country/Region'])).intersection(set(policies.entity)).intersection(set(indicators['Country']))))
    confirmed = pre_series(confirmed, countries)
    deaths = pre_series(deaths, countries)
    recovered = pre_series(recovered, countries)

    # mortality/recovery rate 死亡率恢复率
    mortality = (deaths / confirmed).dropna(how='all').stack().reset_index()
    recovery = (recovered / confirmed).dropna(how='all').stack().reset_index()
    mortality.columns = ['date', 'country', 'y']
    recovery.columns = ['date', 'country', 'y']

    def policy_indicators(policies, rates):
        rates = mortality.copy()
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
            model_importance = model.tree_.compute_feature_importances(normalize=True)
            importance = pd.DataFrame({'Indicator': train_columns, 'importance': model_importance}).sort_values(
                'importance', ascending=False)
            importance = importance[importance['Indicator'] != policy_name]
            importance['Policy'] = policy_name
            res = pd.concat((res, importance[['Policy', 'Indicator', 'importance']][:10]))
        return res

    policy_indicators(policies, recovery).to_csv('output/step_two/policy_indicator.csv', index=False)

if __name__ == '__main__':
    indicator_year = 2019
    indicators = pd.read_excel('raw_data/indicators_all_countries.xlsx')[
        ['Country', 'Indicator', 'Unit', indicator_year]]
    policies = pd.read_excel('raw_data/policies.xlsx')
    policies_all_countries = pd.read_excel('raw_data/policies_all_countries.xlsx')

    # 'confirmed', 'deaths', 'recovered'
    confirmed = pd.read_csv('raw_data/COVID/time_series_covid19_' + 'confirmed' + '_global.csv')
    deaths = pd.read_csv('raw_data/COVID/time_series_covid19_' + 'deaths' + '_global.csv')
    recovered = pd.read_csv('raw_data/COVID/time_series_covid19_' + 'recovered' + '_global.csv')

    # indicator_analysis(confirmed, deaths, recovered, indicators)
    # policy_analysis(confirmed, deaths, recovered, policies_all_countries)
    # policy_country_analysis(confirmed, deaths, recovered, policies)
    # policy_indicator_analysis(confirmed, deaths, recovered, policies_all_countries, indicators)