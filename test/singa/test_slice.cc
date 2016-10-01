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

#include "../src/model/layer/slice.h"
#include "gtest/gtest.h"
#include "singa/singa_config.h"

using singa::Slice;
using singa::Shape;
TEST(Slice, Setup) {
  Slice slice;
  // EXPECT_EQ("PReLU", prelu.layer_type());

  singa::LayerConf conf;
  singa::SliceConf *sliceconf = conf.mutable_slice_conf();
  sliceconf->set_axis(0);
  sliceconf->add_slice_point(2);
  sliceconf->add_slice_point(4);
  slice.Setup({1}, conf);

  EXPECT_EQ(0u, slice.axis());
  EXPECT_EQ(2u, slice.slice_point().size());
  EXPECT_EQ(4u, slice.slice_point().at(1));
}

TEST(Slice, ForwardCPU) {
  const float x[] = {1.f,  2.f, 3.f, 4.f, 5.f, 6.f,
                     -1.f, -2.f, -3.f, -4.f, -5.f, -6.f};
  size_t n = sizeof(x) / sizeof(float);
  size_t batchsize = 6, c = 1, h = 2, w = 1;
  singa::Tensor in(singa::Shape{batchsize, h, w, c});
  in.CopyDataFromHostPtr<float>(x, n);

  Slice slice;
  singa::LayerConf conf;
  singa::SliceConf *sliceconf = conf.mutable_slice_conf();
  sliceconf->set_axis(0);
  sliceconf->add_slice_point(2);
  sliceconf->add_slice_point(4);
  slice.Setup(Shape{h, w, c}, conf);

  vector<singa::Tensor> out = slice.Forward(singa::kTrain, {in});
  EXPECT_EQ(3u, out.size());

  for (size_t i = 0; i < out.size(); i++) {
    const float *yptr = out.at(i).data<float>();
    for (size_t k = 0; k < 2; k++)
      for (size_t j = 0; j < 2; j++)
        EXPECT_FLOAT_EQ(yptr[k*2+j], x[i*2+k*6+j]);
  }
}

TEST(Slice, BackwardCPU) {
  const float x[] = {1.f,  2.f, 3.f, -2.f, -3.f, -1.f,
                     -1.f, 2.f, -1.f, -2.f, -2.f, -1.f};
  size_t n = sizeof(x) / sizeof(float);
  size_t batchsize = 6, c = 1, h = 2, w = 1;
  singa::Tensor in(singa::Shape{batchsize, h, w, c});
  in.CopyDataFromHostPtr<float>(x, n);

  Slice slice;
  singa::LayerConf conf;
  singa::SliceConf *sliceconf = conf.mutable_slice_conf();
  sliceconf->set_axis(0);
  sliceconf->add_slice_point(2);
  sliceconf->add_slice_point(4);
  slice.Setup(Shape{h, w, c}, conf);

  vector<singa::Tensor> out = slice.Forward(singa::kTrain, {in});
  EXPECT_EQ(3u, out.size());

  auto grad = slice.Backward(singa::kTrain, out);
  vector<singa::Tensor> in_grad = grad.first;
  EXPECT_EQ(n, in_grad.at(0).Size());
  const float *yptr = in_grad.at(0).data<float>();

  for (size_t i = 0; i < n; i++)
    EXPECT_FLOAT_EQ(yptr[i], x[i]);
}
