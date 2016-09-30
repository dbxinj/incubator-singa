/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#include "singa/model/loss.h"
#include "gtest/gtest.h"
#include "singa/core/tensor.h"
#include "singa/core/device.h"
#include <math.h>

using singa::Tensor;
class TestTriplet : public::testing::Test {
 protected:
  virtual void SetUp() {
    a.Reshape(singa::Shape{2, 3});
    p.Reshape(singa::Shape{2, 3});
    n.Reshape(singa::Shape{2, 3});
    a.CopyDataFromHostPtr(adat, sizeof(adat) / sizeof(float));
    p.CopyDataFromHostPtr(pdat, sizeof(pdat) / sizeof(float));
    n.CopyDataFromHostPtr(ndat, sizeof(ndat) / sizeof(float));
  }
  const float adat[6] = {0.1, 0.6, 1.5, 2.4, 1.1, 0.5};
  const float pdat[6] = {0.2, 0.4, 0.5, 2.0, 1.2, 0.7};
  const float ndat[6] = {0.8, 0.1, 1.9, 1.5, 1.1, 0.5};

  singa::Tensor a, p, n;
};

TEST_F(TestTriplet, Forward) {
  singa::TripletLoss trip;

  singa::LayerConf conf;
  singa::TripletLossConf* triplet_loss_conf = conf.mutable_triplet_loss_conf();
  triplet_loss_conf->set_margin(1.0);
  trip.Setup(conf);
  EXPECT_EQ(1.0, trip.margin());

  const Tensor loss = trip.Forward(singa::kEval, {a, p, n});
  auto ldat = loss.data<float>();

  for (size_t i = 0; i < loss.Size(); i++) {
    float l = 0.f;
    for (size_t j = 0; j < a.Size() / loss.Size(); j++) {
      size_t idx = i * a.Size() / loss.Size() + j;
      l += (adat[idx] - pdat[idx]) * (adat[idx] - pdat[idx]) -
          (adat[idx] - ndat[idx]) * (adat[idx] - ndat[idx]);
    }
    l = std::max(0.f, 1.f + l);
    EXPECT_FLOAT_EQ(ldat[i], l);
  }
}

TEST_F(TestTriplet, Backward) {
  singa::TripletLoss trip;

  singa::LayerConf conf;
  singa::TripletLossConf* triplet_loss_conf = conf.mutable_triplet_loss_conf();
  triplet_loss_conf->set_margin(1.0);
  trip.Setup(conf);
  EXPECT_EQ(1.0, trip.margin());

  const Tensor loss = trip.Forward(singa::kEval, {a, p, n});
  auto mask = trip.mask();
  auto mdat = mask.data<float>();
  for (size_t i = 0; i < mask.Size(); i++)
    EXPECT_FLOAT_EQ(mdat[i], 1.f);

  const vector<Tensor> grads = trip.Backward(singa::kEval, loss);
  auto da = grads.at(0).data<float>();
  auto dp = grads.at(1).data<float>();
  auto dn = grads.at(2).data<float>();

  float* diff_ap = new float[6];
  float* diff_an = new float[6];
  float* diff_np = new float[6];
  for (size_t i = 0; i < a.Size(); i++) {
    diff_ap[i] = adat[i] - pdat[i];
    diff_an[i] = adat[i] - ndat[i];
    diff_np[i] = ndat[i] - pdat[i];
  }

  for (size_t i = 0; i < a.Size(); i++) {
    EXPECT_FLOAT_EQ(da[i], diff_np[i] * 2.f);
    EXPECT_FLOAT_EQ(dp[i], diff_ap[i] * (-2.f));
    EXPECT_FLOAT_EQ(dn[i], diff_an[i] * 2.f);
  }
}
