import pandas as pd
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.mx.distribution import StudentTOutput
from model.my_network import MyProbEstimator
from preprocess.get_data import get_train_data


def main(series_category='confirmed'):
    # 'confirmed', 'deaths', 'recovered'
    metadata, history_series, train_ds, test_ds, target_scaler, countries = get_train_data(series_category=series_category)

    estimator = MyProbEstimator(
        prediction_length=metadata['prediction_length'],
        context_length=metadata['context_length'],
        freq=metadata['freq'],
        distr_output=StudentTOutput(),
        num_cells=[8, 8, 64],  # [static dim, dynamic dim, target dim]
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
    tss = list(ts_it)
    # sampling
    forecasts = list(forecast_it)

    history_series.to_csv('output/step_one/'+series_category+'/history.csv')  # 存历史数据
    predict_dict = {
        'median': 0.5,
        '35quantile': 0.35,
        '65quantile': 0.65,
        '10quantile': 0.1,
        '90quantile': 0.9,
    }
    history_series=history_series.iloc[:-10]    # 测试效果
    for key in predict_dict.keys():
        median = []
        for i, country in enumerate(countries):
            median.append(forecasts[i].quantile_ts(predict_dict[key]))
        median = pd.concat(median, axis=1)
        median.columns = countries
        median.loc[:, :] = target_scaler.inverse_transform(median).astype(int)
        median.loc[history_series.index[-1]] = history_series.iloc[-1]
        median = median.sort_index().cumsum().iloc[1:]
        median.to_csv('output/step_one/'+series_category+'/' + key + '.csv')


if __name__ == '__main__':
    main(series_category='confirmed')

