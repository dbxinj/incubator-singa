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

#include "singa/model/loss.h"

namespace singa {

void TripletLoss::Setup(const LayerConf &conf) {
  TripletLossConf triplet_loss_conf = conf.triplet_loss_conf();
  margin_ = triplet_loss_conf.margin();
}

Tensor TripletLoss::Forward(int flag, const vector<Tensor>& inputs) {
  Tensor diff_ap, diff_an, diff_np;
  size_t samples = inputs.at(0).shape(0);
  Tensor dist_pos(Shape{samples}), dist_neg(Shape{samples});
  CHECK_EQ(inputs.at(0).shape(0), inputs.at(1).shape(0));
  CHECK_EQ(inputs.at(1).shape(0), inputs.at(2).shape(0));
  diff_ap = inputs.at(0) - inputs.at(1);
  diff_an = inputs.at(0) - inputs.at(2);
  diff_np = inputs.at(2) - inputs.at(1);
  buf_.push(diff_ap);
  buf_.push(diff_an);
  buf_.push(diff_np);
  SumColumns(Square(diff_ap), &dist_pos);
  SumColumns(Square(diff_an), &dist_neg);
  buf_.push((dist_pos - dist_neg + margin_) > 0.f);
  return ReLU(dist_pos - dist_neg + margin_);
}

vector<Tensor> TripletLoss::Backward(int flag, const Tensor& grads) {
  vector<Tensor> in_grads;
  Tensor mask = buf_.top();
  buf_.pop();
  Tensor diff_np = buf_.top();
  buf_.pop();
  Tensor diff_an = buf_.top();
  buf_.pop();
  Tensor diff_ap = buf_.top();
  buf_.pop();
  MultColumn(mask, &diff_np);
  MultColumn(mask, &diff_ap);
  MultColumn(mask, &diff_an);
  in_grads.push_back(diff_np * 2.f);
  in_grads.push_back(diff_ap * (-2.f));
  in_grads.push_back(diff_an * 2.f);
  return in_grads;
}
/*
Tensor TripletLoss::Forward(int flag, const Tensor& prediction, const Tensor& target) {
  CHECK(buf_.empty()) << "Do not call Forward successively for more than twice."
                      << " The calling pattern is [Forward|Evaluate] Backward";
  Tensor t = prediction - target;
  size_t batchsize = 1;
  if (t.nDim() > 1) batchsize = t.shape().at(0);
  size_t dim = t.Size() / batchsize;
  t.Reshape(Shape{batchsize, dim});
  if (kTrain & flag)
    buf_.push(t);
  // TODO(wangwei) use CastType for operator/
  return Sum(Square(t), 1) * 0.5f;
}

Tensor TripletLoss::Backward() {
  Tensor ret = buf_.top();
  buf_.pop();
  return ret * (1.0f / ret.shape().at(0));
}*/
}  // namespace singa
