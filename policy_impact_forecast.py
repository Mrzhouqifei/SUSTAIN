import pandas as pd
import numpy as np
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.mx.distribution import StudentTOutput
from model.my_network import MyProbEstimator
from preprocess.get_data import get_train_data
from preprocess.download_data import download_policy_indicator

def main(series_category, indicator_year):
    # 'confirmed', 'deaths', 'recovered'
    metadata, history_series, train_ds, test_ds, target_scaler, countries = get_train_data(series_category=series_category,
                                                                                           indicator_year=indicator_year)

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
    history_series.to_csv('output/step_three/'+series_category+'/history.csv')  # save history data
    predict_dict = {
        'prediction': 0.5,
    }

    for key in predict_dict.keys():
        median = []
        for i, country in enumerate(countries):
            median.append(forecasts[i].quantile_ts(predict_dict[key]))
        median = pd.concat(median, axis=1)
        median.columns = countries
        median.loc[:, :] = np.maximum(target_scaler.inverse_transform(median).astype(int), 0)
        median.loc[history_series.index[-1]] = history_series.iloc[-1]

        median = median.sort_index().cumsum().iloc[1:]
        median.to_csv('output/step_three/'+series_category+'/' + key + '.csv')


def result_read(series_category):
    history = pd.read_csv('output/step_three/' + series_category + '/' + 'history.csv', index_col=0)
    median = pd.read_csv('output/step_three/' + series_category + '/' + 'prediction.csv', index_col=0)

    history.index = pd.to_datetime(history.index)
    median.index = pd.to_datetime(median.index)

    res_dict = {}
    res_dict['history'] = history
    res_dict['prediction'] = median
    return res_dict


def calculate_rate(numerator, denominator, res_name):
    """
    :param numerator: 分子
    :param denominator: 分母
    :return:
    """
    numerator_dict = result_read(numerator)
    denominator_dict = result_read(denominator)
    names = ['history', 'prediction']
    for name in names:
        tmp = numerator_dict[name] / denominator_dict[name]
        tmp.to_csv('output/step_three/' + res_name + '/' + name + '.csv')


if __name__ == '__main__':
    # update indicator and policy

    # 'confirmed', 'deaths', 'recovered'
    indicator_year = 2019
    main(series_category='confirmed', indicator_year=indicator_year)
    main(series_category='deaths', indicator_year=indicator_year)
    main(series_category='recovered', indicator_year=indicator_year)

    # mortality/recovery rate 死亡率恢复率
    calculate_rate('deaths', 'confirmed', 'mortality')
    calculate_rate('recovered', 'confirmed', 'recovery')

    # pip install pipreqs
    # pipreqs . --encoding=utf8 --force

