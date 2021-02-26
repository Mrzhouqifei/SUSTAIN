import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset


def get_target(series, countries):
    # series (should def a function)，最后添加place holding用来预测

    process_series = series.groupby('Country/Region').sum().drop(['Lat', 'Long'], axis=1).loc[countries, :].T[
        countries].fillna(0)
    process_series.index = pd.to_datetime(process_series.index)
    process_series_diff = process_series.diff()[1:]  # 预测增量

    # define the parameters of the dataset
    metadata = {'num_series': len(countries),
                          'num_steps': len(process_series_diff),
                          'prediction_length': 10,
                          'context_length': 10,
                          'freq': '1D',
                          'start': [pd.Timestamp(process_series_diff.index[0], freq='1D')
                                    for _ in range(len(countries))]
                          }
    # scaler， 对每一个国家的不同日期进行sacle，column=国家
    target_scaler = StandardScaler().fit(process_series_diff.iloc[:-metadata['prediction_length']])
    # 在这里构造代预测的数据
    # num_series, series length
    target = target_scaler.transform(process_series_diff).T

    return metadata, target_scaler, target, process_series, process_series_diff


def get_static(indicators, countries, indicator_year):
    # static covariate (indicators)
    # num_series, indicator num
    indicators_values = indicators.pivot_table(index='Indicator', columns='Country', values=indicator_year)[countries].fillna(0).T
    # 对每一个indicator不同国家的值进行scale
    covariate_s = MinMaxScaler().fit_transform(indicators_values)
    return covariate_s


def get_dynamic(policy, policy_names, countries, process_series_diff):
    # dynamic covariate（time feature and policies）
    # num_series, num policies, series length + (prediction length)
    covariate_d = []
    for policy_name in policy_names:
        tmp = policy[['entity', 'date', policy_name]].pivot_table(index='date', columns='entity', values=policy_name)
        for x in countries:
            if x not in tmp.columns:
                tmp[x] = 0
        tmp = tmp[countries]
        tmp.index = pd.to_datetime(tmp.index)

        for index in process_series_diff.index:
            if index not in tmp.index:
                tmp.loc[index] = 0

        tmp = tmp.sort_index()
        tmp = tmp.loc[process_series_diff.index[0]:process_series_diff.index[-1]].fillna(0)
        #     tmp_value = MinMaxScaler().fit_transform(tmp).T
        tmp_value = tmp.T.values
        covariate_d.append(tmp_value)
    covariate_d = np.array(covariate_d).transpose(1, 0, 2)
    return covariate_d


def get_train_data(
        series_category: str = 'confirmed',
        indicator_year: int = 2019,
):
    """
    :param series_category: 'confirmed', 'deaths', 'recovered'
    :param indicator_year:
    :return:
    """
    # 拿2019年的indicators
    indicators = pd.read_excel('raw_data/SUSTAIN database_08Jan2021_Asia and Latin America.xlsx')[
        ['Country', 'Indicator', indicator_year]]
    policies = pd.read_excel('raw_data/SUSTAIN database_09Jan2021_policies_Asia and Latin America.xlsx')

    if series_category == 'deaths':
        series = pd.read_csv('raw_data/COVID/time_series_covid19_deaths_global.csv')
    elif series_category == 'recovered':
        series = pd.read_csv('raw_data/COVID/time_series_covid19_recovered_global.csv')
    else:
        series = pd.read_csv('raw_data/COVID/time_series_covid19_confirmed_global.csv')

    countries = sorted(list(
        set(policies.entity).intersection(set(series['Country/Region'])).intersection(set(indicators['Country']))))
    indicator_names = list(set(indicators.Indicator))
    policy_names = [x for x in list(policies.columns) if x not in ['entity', 'iso', 'date']]

    metadata, target_scaler, target, process_series, process_series_diff = get_target(series, countries)
    covariate_s = get_static(indicators, countries, indicator_year=indicator_year)
    covariate_d = get_dynamic(policies, policy_names, countries, process_series_diff)

    train_ds = ListDataset([{FieldName.TARGET: target,
                             FieldName.START: start,
                             FieldName.FEAT_STATIC_REAL: fsr,
                             FieldName.FEAT_DYNAMIC_REAL: fdr,
                             }
                            for (target, start, fsr, fdr) in zip(target[:, :-metadata['prediction_length']],
                                                                 metadata['start'],
                                                                 covariate_s,
                                                                 covariate_d[:, :,
                                                                 :-metadata['prediction_length']],
                                                                 )],
                           freq=metadata['freq'])

    test_ds = ListDataset([{FieldName.TARGET: target,
                            FieldName.START: start,
                            FieldName.FEAT_STATIC_REAL: fsr,
                            FieldName.FEAT_DYNAMIC_REAL: fdr,
                            }
                           for (target, start, fsr, fdr) in zip(target,
                                                                metadata['start'],
                                                                covariate_s,
                                                                covariate_d,
                                                                )],
                          freq=metadata['freq'])

    return metadata, process_series, train_ds, test_ds, target_scaler, countries






