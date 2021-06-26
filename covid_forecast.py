import mxnet.random
import pandas as pd
import numpy as np
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.mx.distribution import StudentTOutput
from model.my_network import MyProbEstimator
from preprocess.get_data import get_train_data
from update_data import covid_update
from preprocess.download_data import download_policy_indicator

def main(series_category, indicator_year, predict_len, type='business'):
    res_prefix = 'covid_forecast' if type == 'business' else 'covid_forecast_with_future_policy'
    # 'confirmed', 'deaths', 'recovered'
    metadata, history_series, train_ds, test_ds, target_scaler, countries = get_train_data(type=type,
                                                                                           series_category=series_category,
                                                                                           indicator_year=indicator_year,
                                                                                           predict_len=predict_len)

    estimator = MyProbEstimator(
        prediction_length=metadata['prediction_length'],
        context_length=metadata['context_length'],
        freq=metadata['freq'],
        distr_output=StudentTOutput(),
        num_cells=[8, 16, 64],  # [static dim, dynamic dim, target dim]
        scaling=True,
        trainer=Trainer(ctx="cpu",  # gpu(0),
                        epochs=10,
                        learning_rate=1e-3,
                        num_batches_per_epoch=100,
                        batch_size=32,
                        hybridize=False,
                       )
    )

    predictor = estimator.train(train_ds)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )
    # tss = list(ts_it)
    # sampling
    forecasts = list(forecast_it)

    history_series = history_series.iloc[:-metadata['prediction_length']]  # filter placehold data
    history_series.to_csv('output/' + res_prefix + '/'+series_category+'/history.csv')  # save history data
    predict_dict = {
        'prediction_median': 0.5,
        'prediction_lower': 0.35,
        'prediction_upper': 0.65,
        # 'quantile10': 0.1,
        # 'quantile90': 0.9,
    }

    for key in predict_dict.keys():
        median = []
        for i, country in enumerate(countries):
            median.append(forecasts[i].quantile_ts(predict_dict[key]))
        median = pd.concat(median, axis=1)
        median.columns = countries
        # 'confirmed', 'deaths', 'recovered'
        if series_category == 'contagion' or \
                series_category == 'confirmed_number' or \
                series_category == 'dead_number' or \
                series_category == 'recovered_number':
            median.loc[:, :] = np.maximum(median, 0)
        # 比例
        # median.loc[history_series.index[-1]] = history_series.diff().iloc[-1]
        # median = median.sort_index().cumprod().iloc[1:]  #

        median.loc[history_series.index[-1]] = history_series.iloc[-1]
        median = median.sort_index().cumsum().iloc[1:]  #
        median.to_csv('output/' + res_prefix + '/'+series_category+'/' + key + '.csv')


def result_read(series_category, type='business'):
    res_prefix = 'covid_forecast' if type == 'business' else 'covid_forecast_with_future_policy'
    history = pd.read_csv('output/' + res_prefix + '/' + series_category + '/' + 'history.csv', index_col=0)
    median = pd.read_csv('output/' + res_prefix + '/' + series_category + '/' + 'prediction_median.csv', index_col=0)
    # quantile10 = pd.read_csv('output/covid_forecast/' + series_category + '/' + 'quantile10.csv', index_col=0)
    # quantile90 = pd.read_csv('output/covid_forecast/' + series_category + '/' + 'quantile90.csv', index_col=0)
    quantile35 = pd.read_csv('output/' + res_prefix + '/' + series_category + '/' + 'prediction_lower.csv', index_col=0)
    quantile65 = pd.read_csv('output/' + res_prefix + '/' + series_category + '/' + 'prediction_upper.csv', index_col=0)

    history.index = pd.to_datetime(history.index)
    median.index = pd.to_datetime(median.index)
    # quantile10.index = pd.to_datetime(quantile10.index)
    # quantile90.index = pd.to_datetime(quantile90.index)
    quantile35.index = pd.to_datetime(quantile35.index)
    quantile65.index = pd.to_datetime(quantile65.index)
    res_dict = {}
    res_dict['history'] = history.fillna(0)
    res_dict['prediction_median'] = median.fillna(0)
    # res_dict['quantile10'] = quantile10
    # res_dict['quantile90'] = quantile90
    res_dict['prediction_lower'] = quantile35.fillna(0)
    res_dict['prediction_upper'] = quantile65.fillna(0)
    return res_dict


def calculate_rate(numerator, denominator, res_name):
    """
    :param numerator: 分子
    :param denominator: 分母
    :return:
    """
    numerator_dict = result_read(numerator)
    denominator_dict = result_read(denominator)
    names = ['history', 'quantile10', 'quantile35', 'median', 'quantile65', 'quantile90']
    for name in names[:1]:  # history
        tmp = (numerator_dict[name] / denominator_dict[name]).replace([float('inf'), np.nan]).dropna(how='all')
        tmp.to_csv('output/covid_forecast/' + res_name + '/' + name + '.csv')

    columns = list(numerator_dict['median'].columns)
    index = list(numerator_dict['median'].index)
    res = []
    for name in names[1:]:
        res.append((numerator_dict[name] / denominator_dict[name]).replace([float('inf'), np.nan]).dropna(how='all').values)
    res = np.sort(np.array(res), axis=0)

    for i, name in enumerate(names[1:]):  # future prediction
        tmp = pd.DataFrame(res[i, :, :], columns=columns)
        tmp.index = index
        tmp.to_csv('output/covid_forecast/' + res_name + '/' + name + '.csv')


def calculate_confirmed_number(type='business'):
    res_prefix = 'covid_forecast' if type == 'business' else 'covid_forecast_with_future_policy'
    population = pd.read_excel('raw_data/PopulationData.xls', skiprows=[0, 1, 2])[['Country', population_year]]
    population[population_year] = population[population_year] * 1000
    population = population.set_index('Country').T
    numerator_dict = result_read('contagion', type)

    intersection_country = list(set(numerator_dict['history'].columns).intersection(population.columns))
    population = population[intersection_country]

    names = ['history', 'prediction_lower', 'prediction_median', 'prediction_upper']
    for name in names:
        numerator_dict[name] = numerator_dict[name][intersection_country]
        confirmed_number = (numerator_dict[name].values * population.values).astype(int)
        confirmed_number = pd.DataFrame(confirmed_number, columns=[intersection_country], index=numerator_dict[name].index)
        confirmed_number.to_csv('output/' + res_prefix + '/confirmed_number/' + name + '.csv')


def calculate_dead_recovered_number(target, type='business'):
    res_prefix = 'covid_forecast' if type == 'business' else 'covid_forecast_with_future_policy'
    confirmed_number = result_read('confirmed_number')
    if target == 'dead':
        mortality = result_read('mortality', type)
        res = 'dead_number/'
    elif target == 'recovered':
        mortality = result_read('recovery', type)
        res = 'recovered_number/'

    intersection_country = list(set(confirmed_number['history'].columns).intersection(set(mortality['history'].columns)))

    names = ['history', 'prediction_lower', 'prediction_median', 'prediction_upper']
    for name in names:
        confirmed_number[name] = confirmed_number[name][intersection_country]
        mortality[name] = mortality[name][intersection_country]

        dead_number = (confirmed_number[name].values * mortality[name].values).astype(int)
        dead_number = pd.DataFrame(dead_number, columns=[intersection_country], index=confirmed_number[name].index)
        if name == 'history':
            for column in dead_number.columns:
                dead_number[column] = np.sort(dead_number[column])
            tmp = dead_number.iloc[-1, :]
        else:
            for column in dead_number.columns:
                dead_number[column] = np.maximum(dead_number[column], tmp[column])
                dead_number[column] = np.sort(dead_number[column])
        dead_number.to_csv('output/' + res_prefix + '/' + res + name + '.csv')


def contagion(res_name, population_year):
    population = pd.read_excel('raw_data/PopulationData.xls', skiprows=[0, 1, 2])[['Country', population_year]]
    population[population_year] = population[population_year] * 1000
    population = population.set_index('Country').T
    numerator_dict = result_read('confirmed')

    intersection_country = list(set(numerator_dict['history'].columns).intersection(population.columns))
    population = population[intersection_country]

    names = ['history', 'quantile10', 'quantile35', 'median', 'quantile65', 'quantile90']
    for name in names:
        numerator_dict[name] = numerator_dict[name][intersection_country]
        contagion = numerator_dict[name].values / population.values
        contagion = pd.DataFrame(contagion, columns=[intersection_country], index=numerator_dict[name].index)
        contagion.to_csv('output/covid_forecast/' + res_name + '/' + name + '.csv')


def calculate_contagion_with_popultation(series_category, target, population_year):
    series = pd.read_csv('raw_data/COVID/time_series_covid19_' + series_category + '_global.csv')
    process_series = series.groupby('Country/Region').sum().drop(['Lat', 'Long'], axis=1).T.fillna(0)
    process_series.index = pd.to_datetime(process_series.index)

    population = pd.read_excel('raw_data/PopulationData.xls', skiprows=[0, 1, 2])[['Country', population_year]]
    population[population_year] = population[population_year] * 1000
    population = population.set_index('Country').T

    intersection_country = list(set(process_series.columns).intersection(population.columns))
    population = population[intersection_country]
    process_series = process_series[intersection_country]

    target_series = pd.DataFrame(process_series.values / population.values, columns=[intersection_country],
                                 index=process_series.index)
    target_series.to_csv('raw_data/COVID/time_series_covid19_' + target + '_global.csv')


def cal_raw_rate(series_category, target):
    confirmed = pd.read_csv('raw_data/COVID/time_series_covid19_confirmed_global.csv')
    confirmed_series = confirmed.groupby('Country/Region').sum().drop(['Lat', 'Long'], axis=1).T.fillna(0)
    confirmed_series.index = pd.to_datetime(confirmed_series.index)

    series = pd.read_csv('raw_data/COVID/time_series_covid19_' + series_category + '_global.csv')
    process_series = series.groupby('Country/Region').sum().drop(['Lat', 'Long'], axis=1).T.fillna(0)
    process_series.index = pd.to_datetime(process_series.index)
    intersection_country = list(set(process_series.columns).intersection(confirmed_series.columns))
    target_series = process_series[intersection_country] / confirmed_series[intersection_country]
    target_series.to_csv('raw_data/COVID/time_series_covid19_' + target + '_global.csv')


def process_number_series(series_category):
    confirmed = pd.read_csv('raw_data/COVID/time_series_covid19_confirmed_global.csv')
    confirmed_series = confirmed.groupby('Country/Region').sum().drop(['Lat', 'Long'], axis=1).T.fillna(0)
    confirmed_series.index = pd.to_datetime(confirmed_series.index)

    series = pd.read_csv('raw_data/COVID/time_series_covid19_' + series_category + '_global.csv')
    process_series = series.groupby('Country/Region').sum().drop(['Lat', 'Long'], axis=1).T.fillna(0)
    process_series.index = pd.to_datetime(process_series.index)
    if series_category == 'deaths':
        series_category = 'dead'
    process_series.to_csv('raw_data/COVID/time_series_covid19_' + series_category + '_number' + '_global.csv')


def business_as_usual(indicator_year, population_year, predict_len):
    calculate_contagion_with_popultation('confirmed', 'contagion', population_year)
    cal_raw_rate('deaths', 'mortality')
    cal_raw_rate('recovered', 'recovery')

    main(series_category='contagion', indicator_year=indicator_year, predict_len=predict_len)
    main(series_category='mortality', indicator_year=indicator_year, predict_len=predict_len)
    main(series_category='recovery', indicator_year=indicator_year, predict_len=predict_len)
    calculate_confirmed_number()
    main(series_category='dead_number', indicator_year=indicator_year, predict_len=predict_len)
    main(series_category='recovered_number', indicator_year=indicator_year, predict_len=predict_len)

    # calculate_dead_recovered_number('dead')
    # calculate_dead_recovered_number('recovered')
    # main(series_category='confirmed_number', indicator_year=indicator_year, predict_len=predict_len)



    # # 'confirmed', 'deaths', 'recovered'
    # main(series_category='confirmed', indicator_year=indicator_year, predict_len=predict_len)
    # main(series_category='deaths', indicator_year=indicator_year, predict_len=predict_len)
    # main(series_category='recovered', indicator_year=indicator_year, predict_len=predict_len)
    #
    # # mortality/recovery rate 死亡率恢复率
    # calculate_rate('deaths', 'confirmed', 'mortality')
    # calculate_rate('recovered', 'confirmed', 'recovery')
    #
    # # contagion rate 传染率
    # contagion('contagion', population_year=population_year)


def covid_forecast_with_future_policy(indicator_year, population_year, predict_len):
    calculate_contagion_with_popultation('confirmed', 'contagion', population_year)
    cal_raw_rate('deaths', 'mortality')
    cal_raw_rate('recovered', 'recovery')

    main(series_category='contagion', indicator_year=indicator_year, predict_len=predict_len, type='policy')
    main(series_category='mortality', indicator_year=indicator_year, predict_len=predict_len, type='policy')
    main(series_category='recovery', indicator_year=indicator_year, predict_len=predict_len, type='policy')
    calculate_confirmed_number(type='policy')
    main(series_category='dead_number', indicator_year=indicator_year, predict_len=predict_len, type='policy')
    main(series_category='recovered_number', indicator_year=indicator_year, predict_len=predict_len, type='policy')


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
        'C1_School closing': 0.998,
#         'H4_Emergency investment in healthcare': 0.992,
#         'E3_Fiscal measures': 0.995,
#         'E4_International support': 0.998,
#         'H5_Investment in vaccines': 0.999
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
        print(delta_cg)
        delta_y = [delta_cg * x for x in delta_y]

    mixed_policy_series = []
    pre_y = newest_real_contagion
    for item in delta_y:
        y = pre_y + item
        mixed_policy_series.append(y)
        pre_y = y
    return mixed_policy_series


if __name__ == '__main__':
    init_seed = 2020
    np.random.seed(init_seed)
    mxnet.random.seed(init_seed)

    # covid_update()  # update data
    indicator_year, population_year = 2019, 2020
    predict_len = 30
    # 'confirmed', 'deaths', 'recovered'
    # process_number_series('confirmed')

    process_number_series('deaths')
    process_number_series('recovered')
    business_as_usual(indicator_year, population_year, predict_len)
    # covid_forecast_with_future_policy(indicator_year, population_year, predict_len)

    # pip install pipreqs
    # pipreqs . --encoding=utf8 --force

