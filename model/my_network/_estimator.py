from typing import List, Tuple
# from gluonts.mx import Tensor

from gluonts.mx.block.scaler import MeanScaler, NOPScaler
from gluonts.model.estimator import GluonEstimator
from gluonts.mx.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.core.component import validated
from gluonts.mx.trainer import Trainer
from gluonts.support.util import copy_parameters
from gluonts.transform import ExpectedNumInstanceSampler, Transformation, InstanceSplitter
from gluonts.mx.distribution.distribution_output import DistributionOutput
from mxnet.gluon import HybridBlock
from gluonts.dataset.field_names import FieldName
from mxnet import gluon
import mxnet as mx
from gluonts.transform import (
    AddAgeFeature,
    AddTimeFeatures,
    AddObservedValuesIndicator,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    SetFieldIfNotPresent,
    AsNumpyArray,
    VstackFeatures,
)
from gluonts.support.util import weighted_average

from ._network import (
    MyProbNetwork,
    MyProbTrainNetwork,
    MyProbPredNetwork,
)

from gluonts.time_feature import (
    TimeFeature,
    get_lags_for_frequency,
    time_features_from_frequency_str,
)

"""
    START = "start"
    TARGET = "target"

    FEAT_STATIC_CAT = "feat_static_cat"
    FEAT_STATIC_REAL = "feat_static_real"
    FEAT_DYNAMIC_CAT = "feat_dynamic_cat"
    FEAT_DYNAMIC_REAL = "feat_dynamic_real"
    PAST_FEAT_DYNAMIC_REAL = "past_feat_dynamic_real"
    FEAT_DYNAMIC_REAL_LEGACY = "dynamic_feat"

    FEAT_DYNAMIC = "feat_dynamic"
    PAST_FEAT_DYNAMIC = "past_feat_dynamic"

    FEAT_DYNAMIC = "feat_dynamic"

    FEAT_TIME = "time_feat"
    FEAT_CONST = "feat_dynamic_const"
    FEAT_AGE = "feat_dynamic_age"

    OBSERVED_VALUES = "observed_values"
    IS_PAD = "is_pad"
    FORECAST_START = "forecast_start"

    TARGET_DIM_INDICATOR = "target_dimension_indicator"
"""

class MyProbEstimator(GluonEstimator):
    @validated()
    def __init__(
            self,
            prediction_length: int,
            context_length: int,
            freq: str,
            distr_output: DistributionOutput,
            num_cells: List,
            num_sample_paths: int = 100,
            scaling: bool = True,
            trainer: Trainer = Trainer()
    ) -> None:
        super().__init__(trainer=trainer)
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.freq = freq
        self.distr_output = distr_output
        self.num_cells = num_cells
        self.num_sample_paths = num_sample_paths
        self.scaling = scaling
        self.time_features = time_features_from_frequency_str(self.freq)


    def create_transformation(self):
        # Feature transformation that the model uses for input.
        return Chain(
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                    dtype=self.dtype,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_DYNAMIC_REAL,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_DYNAMIC_REAL]
                ),
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    past_length=self.context_length,
                    future_length=self.prediction_length,
                    time_series_fields=[
                        FieldName.FEAT_DYNAMIC_REAL,
                        FieldName.OBSERVED_VALUES,
                    ],
                ),

            ]
        )

    def create_training_network(self) -> MyProbTrainNetwork:
        return MyProbTrainNetwork(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            distr_output=self.distr_output,
            num_cells=self.num_cells,
            num_sample_paths=self.num_sample_paths,
            scaling=self.scaling
        )

    def create_predictor(
            self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_network = MyProbPredNetwork(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            distr_output=self.distr_output,
            num_cells=self.num_cells,
            num_sample_paths=self.num_sample_paths,
            scaling=self.scaling
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )