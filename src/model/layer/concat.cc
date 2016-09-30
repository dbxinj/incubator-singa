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
#include "./concat.h"
namespace singa {

RegisterLayerClass(singa_concat, Concat);

void Concat::Setup(const Shape& in_sample, const LayerConf& conf) {
  Layer::Setup(in_sample, conf);
  ConcatConf concat_conf = conf.concat_conf();
  axis_ = concat_conf.axis();
  out_sample_shape_ = in_sample;
  input_size_ = concat_conf.in_size();
  if (axis_) out_sample_shape_.at(axis_-1) *= input_size_;
}

const vector<Tensor> Concat::Forward(int flag, const vector<Tensor>& inputs) {
  /// currently only support concat inputs with equal size.
  vector<Tensor> outputs;
  if (axis_) out_sample_shape_.at(axis_-1) *= input_size_;
  size_t data_size = inputs.at(0).Size();
  auto dev = inputs.at(0).device();
  Shape shape = inputs.at(0).shape();
  vector<Tensor> in;
  for (size_t i = 0; i < inputs.size(); i++) {
    CHECK_EQ(data_size, inputs.at(i).Size());
    Tensor tmp(shape);
    tmp.CopyData(inputs.at(i));
    tmp.ToHost();
    in.push_back(tmp);
  }
  shape.at(axis_) *= input_size_;
  Tensor tmp(shape);
  for (size_t i = 0u; i < input_size_; i++) {
    const float* data = in.at(i).data<float>();
    tmp.CopyDataFromHostPtr<float>(data, data_size, i * data_size);
  }
  tmp.ToDevice(dev);
  outputs.push_back(tmp);
  return outputs;
}

const std::pair<vector<Tensor>, vector<Tensor>> Concat::Backward(
    int flag, const vector<Tensor>& grads) {
  vector<Tensor> in_grad, param_grad;
  Shape in_shape = grads.at(0).shape();
  auto dev = grads.at(0).device();
  Tensor out_grad(in_shape);
  out_grad.CopyData(grads.at(0));
  out_grad.ToHost();
  in_shape.at(axis_) /= input_size_;
  size_t data_size = out_grad.Size() / input_size_;
  const float* data = out_grad.data<float>();
  for (size_t i = 0u; i < input_size_; i++) {
    Tensor grad(in_shape);
    grad.CopyDataFromHostPtr<float>(data + i * data_size, data_size);
    grad.ToDevice(dev);
    in_grad.push_back(grad);
  }
  return std::make_pair(in_grad, param_grad);
}
}  // namespace singa
