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
    res_prefix = 'covid_forecast' if type == 'business' else 'covid_forecast_policy'
    # 'confirmed', 'deaths', 'recovered'
    metadata, history_series, train_ds, test_ds, target_scaler, countries = get_train_data(series_category=series_category,
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
        if series_category == 'contagion':
            median.loc[:, :] = np.maximum(median, 0)
        # 比例
        # median.loc[history_series.index[-1]] = history_series.diff().iloc[-1]
        # median = median.sort_index().cumprod().iloc[1:]  #

        median.loc[history_series.index[-1]] = history_series.iloc[-1]
        median = median.sort_index().cumsum().iloc[1:]  #
        median.to_csv('output/' + res_prefix + '/'+series_category+'/' + key + '.csv')


def result_read(series_category):
    history = pd.read_csv('output/covid_forecast/' + series_category + '/' + 'history.csv', index_col=0)
    median = pd.read_csv('output/covid_forecast/' + series_category + '/' + 'prediction_median.csv', index_col=0)
    # quantile10 = pd.read_csv('output/covid_forecast/' + series_category + '/' + 'quantile10.csv', index_col=0)
    # quantile90 = pd.read_csv('output/covid_forecast/' + series_category + '/' + 'quantile90.csv', index_col=0)
    quantile35 = pd.read_csv('output/covid_forecast/' + series_category + '/' + 'prediction_lower.csv', index_col=0)
    quantile65 = pd.read_csv('output/covid_forecast/' + series_category + '/' + 'prediction_lower.csv', index_col=0)

    history.index = pd.to_datetime(history.index)
    median.index = pd.to_datetime(median.index)
    # quantile10.index = pd.to_datetime(quantile10.index)
    # quantile90.index = pd.to_datetime(quantile90.index)
    quantile35.index = pd.to_datetime(quantile35.index)
    quantile65.index = pd.to_datetime(quantile65.index)
    res_dict = {}
    res_dict['history'] = history
    res_dict['prediction_median'] = median
    # res_dict['quantile10'] = quantile10
    # res_dict['quantile90'] = quantile90
    res_dict['prediction_lower'] = quantile35
    res_dict['prediction_upper'] = quantile65
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


def business_as_usual(indicator_year, population_year, predict_len):
    calculate_contagion_with_popultation('confirmed', 'contagion', population_year)
    cal_raw_rate('deaths', 'mortality')
    cal_raw_rate('recovered', 'recovery')

    main(series_category='contagion', indicator_year=indicator_year, predict_len=predict_len)
    main(series_category='mortality', indicator_year=indicator_year, predict_len=predict_len)
    main(series_category='recovery', indicator_year=indicator_year, predict_len=predict_len)

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


def covid_forecast_policy(indicator_year, population_year, predict_len):
    calculate_contagion_with_popultation('confirmed', 'contagion', population_year)
    cal_raw_rate('deaths', 'mortality')
    cal_raw_rate('recovered', 'recovery')

    main(series_category='contagion', indicator_year=indicator_year, predict_len=predict_len, type='policy')
    main(series_category='mortality', indicator_year=indicator_year, predict_len=predict_len, type='policy')
    main(series_category='recovery', indicator_year=indicator_year, predict_len=predict_len, type='policy')


if __name__ == '__main__':
    # covid_update()  # update data
    indicator_year, population_year = 2019, 2020
    predict_len = 30
    # business_as_usual(indicator_year, population_year, predict_len)

    covid_forecast_policy(indicator_year, population_year, predict_len)

    # pip install pipreqs
    # pipreqs . --encoding=utf8 --force

