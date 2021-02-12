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
    AddObservedValuesIndicator,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    SetFieldIfNotPresent,
)
from gluonts.support.util import weighted_average

class MyProbNetwork(gluon.HybridBlock):
    def __init__(self,
                 prediction_length,
                 context_length,
                 distr_output,
                 num_cells,
                 num_sample_paths=100,
                 scaling=True,
                 **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.distr_output = distr_output
        self.num_cells = num_cells  # [static dim, dynamic dim, target dim]
        self.num_sample_paths = num_sample_paths
        self.proj_distr_args = distr_output.get_args_proj()
        self.scaling = scaling

        with self.name_scope():
            # # static dense
            self.static_mlp = mx.gluon.nn.HybridSequential()
            self.static_mlp.add(mx.gluon.nn.Dense(units=self.num_cells[0], activation="relu"))
            # dynamic dense
            self.dynamic_mlp = mx.gluon.nn.HybridSequential()
            self.dynamic_mlp.add(mx.gluon.nn.Dense(units=self.num_cells[1], activation="relu"))
            # target mlp
            self.mlp = mx.gluon.nn.HybridSequential()
            self.mlp.add(mx.gluon.nn.Dense(units=self.num_cells[2], activation="relu"))
            self.mlp.add(mx.gluon.nn.Dense(units=prediction_length * self.num_cells[2]))
            self.mlp.add(
                mx.gluon.nn.HybridLambda(
                    lambda F, o: F.reshape(
                        o, (-1, prediction_length, self.num_cells[2])
                    )
                )
            )
            # future mlp
            

            if scaling:
                self.scaler = MeanScaler(keepdims=True)
            else:
                self.scaler = NOPScaler(keepdims=True)

    def compute_scale(self, past_target, past_observed_values):
        # scale shape is (batch_size, 1)
        _, scale = self.scaler(
            past_target.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
            past_observed_values.slice_axis(
                axis=1, begin=-self.context_length, end=None
            ),
        )

        return scale

"""
target time series
dynamic time series
static real
"""
class MyProbTrainNetwork(MyProbNetwork):
    def hybrid_forward(self,
                       F,
                       past_target,
                       future_target,
                       past_observed_values,
                       future_observed_values,
                       feat_static_real,
                       past_feat_dynamic_real,
                       future_feat_dynamic_real
                       ):
        # compute scale
        scale = self.compute_scale(past_target, past_observed_values)

        # scale target and time features
        past_target_scale = F.broadcast_div(past_target, scale)
        # past_feat_dynamic_real_scale = F.broadcast_div(past_feat_dynamic_real.squeeze(axis=-1), scale)

        static_output = self.static_mlp(feat_static_real)
        past_feat_dynamic_real = past_feat_dynamic_real.reshape(past_feat_dynamic_real.shape[0], -1)
        dynamic_output = self.dynamic_mlp(past_feat_dynamic_real)

        # concatenate target and time features to use them as input to the network
        # net_input = F.concat(past_target_scale, past_feat_dynamic_real_scale, dim=-1)
        net_input = F.concat(past_target_scale, static_output, dynamic_output, dim=-1)

        # print(feat_static_real.shape)
        # print(past_target.shape, past_target_scale.shape)
        # print(past_feat_dynamic_real.shape, past_feat_dynamic_real_scale.shape)
        # compute network output
        net_output = self.mlp(net_input)

        # (batch, prediction_length * nn_features)  ->  (batch, prediction_length, nn_features)
        net_output = net_output.reshape(0, self.prediction_length, -1)

        # future mlp

        # project network output to distribution parameters domain
        distr_args = self.proj_distr_args(net_output)

        # compute distribution
        distr = self.distr_output.distribution(distr_args, scale=scale)

        # negative log-likelihood
        loss = distr.loss(future_target)

        weighted_loss = weighted_average(
            F=F, x=loss, weights=future_observed_values, axis=1
        )

        return weighted_loss


class MyProbPredNetwork(MyProbTrainNetwork):
    # The prediction network only receives past_target and returns predictions
    def hybrid_forward(self,
                       F,
                       past_target,
                       past_observed_values,
                       feat_static_real,
                       past_feat_dynamic_real,
                       future_feat_dynamic_real
                       ):
        # repeat fields: from (batch_size, past_target_length) to
        # (batch_size * num_sample_paths, past_target_length)
        repeated_past_target = past_target.repeat(
            repeats=self.num_sample_paths, axis=0
        )
        repeated_past_observed_values = past_observed_values.repeat(
            repeats=self.num_sample_paths, axis=0
        )
        repeated_feat_static_real = feat_static_real.repeat(
            repeats=self.num_sample_paths, axis=0
        )
        repeated_past_feat_dynamic_real = past_feat_dynamic_real.repeat(
            repeats=self.num_sample_paths, axis=0
        )
        repeated_future_feat_dynamic_real = future_feat_dynamic_real.repeat(
            repeats=self.num_sample_paths, axis=0
        )

        # compute scale
        scale = self.compute_scale(repeated_past_target, repeated_past_observed_values)

        # scale repeated target and time features
        repeated_past_target_scale = F.broadcast_div(repeated_past_target, scale)
        # repeated_past_feat_dynamic_real_scale = F.broadcast_div(repeated_past_feat_dynamic_real.squeeze(axis=-1), scale)

        static_output = self.static_mlp(repeated_feat_static_real)
        repeated_past_feat_dynamic_real = repeated_past_feat_dynamic_real.reshape(repeated_past_feat_dynamic_real.shape[0], -1)
        dynamic_output = self.dynamic_mlp(repeated_past_feat_dynamic_real)

        # concatenate target and time features to use them as input to the network
        # net_input = F.concat(repeated_past_target_scale, repeated_past_feat_dynamic_real_scale, dim=-1)
        net_input = F.concat(repeated_past_target_scale, static_output, dynamic_output, dim=-1)

        # compute network oputput
        net_output = self.mlp(net_input)

        # (batch * num_sample_paths, prediction_length * nn_features)  ->  (batch * num_sample_paths, prediction_length, nn_features)
        net_output = net_output.reshape(0, self.prediction_length, -1)

        # project network output to distribution parameters domain
        distr_args = self.proj_distr_args(net_output)

        # compute distribution
        distr = self.distr_output.distribution(distr_args, scale=scale)

        # get (batch_size * num_sample_paths, prediction_length) samples
        samples = distr.sample()

        # reshape from (batch_size * num_sample_paths, prediction_length) to
        # (batch_size, num_sample_paths, prediction_length)
        return samples.reshape(shape=(-1, self.num_sample_paths, self.prediction_length))