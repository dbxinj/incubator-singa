/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "singa/model/layer.h"
#include "./slice.h"
#include "singa/utils/string.h"
namespace singa {

RegisterLayerClass(singa_slice, Slice);

void Slice::Setup(const Shape& in_sample, const LayerConf& conf) {
  Layer::Setup(in_sample, conf);
  SliceConf slice_conf = conf.slice_conf();
  for (int i = 0; i < slice_conf.slice_point_size(); i++)
    slice_point_.push_back(slice_conf.slice_point(i));
  axis_ = slice_conf.axis();
  output_size_ = slice_conf.slice_point_size() + 1;
  out_sample_shape_ = in_sample;
  if (axis_) {
    CHECK_EQ(out_sample_shape_.at(axis_-1) % output_size_, 0);
    out_sample_shape_.at(axis_-1) /= output_size_;
  }
}

const vector<Tensor> Slice::Forward(int flag, const vector<Tensor>& inputs) {
  /// currently only support slice equally.
  vector<Tensor> outputs;
  CHECK_LE(inputs.size(), 1u);
  size_t samples = inputs.at(0).shape(0) / output_size_;
  size_t data_size = inputs.at(0).Size() / inputs.at(0).shape(0);
  Shape shape = inputs.at(0).shape();
  auto dev = inputs.at(0).device();
  Tensor input(shape);
  input.CopyData(inputs.at(0));
  input.ToHost();
  const float* data = input.data<float>();
  shape.at(axis_) /= output_size_;
  for (size_t k = 0; k < output_size_; k++) {
    Tensor tmp(shape);
    for (size_t i = 0; i < samples; i++)
      tmp.CopyDataFromHostPtr<float>(data + (output_size_ * i + k) * data_size,
          data_size);
    tmp.ToDevice(dev);
    outputs.push_back(tmp);
  }
  return outputs;
}

const std::pair<vector<Tensor>, vector<Tensor>> Slice::Backward(
    int flag, const vector<Tensor>& grads) {
  /// currently only support slice equally.
  vector<Tensor> input_grad, param_grad;
  Shape input_shape = grads.at(0).shape();
  size_t samples = grads.at(0).shape(0);
  auto dev = grads.at(0).device();
  vector<Tensor> out_grads;
  for (size_t i = 0; i < grads.size(); i++) {
    Tensor tmp(grads.at(i).shape());
    tmp.CopyData(grads.at(i));
    tmp.ToHost();
    out_grads.push_back(tmp);
  }
  input_shape.at(axis_) *= output_size_;
  Tensor grad(input_shape);
  size_t data_size = out_grads.at(0).Size() / outputs.at(0).shape(0);
  for (size_t k = 0; k < output_size_; k++) {
    const float* data = out_grads.at(k).data<float>();
    for (size_t i = 0; i < samples; i++) {
      grad.CopyDataFromHostPtr<float>(data + i * data_size,
          data_size, (output_size_ * i + k) * data_size);
    }
  }
  grad.ToDevice(dev);
  input_grad.push_back(grad);
  return std::make_pair(input_grad, param_grad);
}
}  // namespace singa
