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
#ifndef SRC_MODEL_LAYER_CONCAT_H_
#define SRC_MODEL_LAYER_CONCAT_H_
#include <string>
#include <utility>
#include <vector>
#include "singa/model/layer.h"

namespace singa {
/// Slice data according to the axis of input layer
class Concat : public Layer {
 public:
   const std::string layer_type() const override { return "Concat"; }

   /// the sample shape of all input tesnors should be the same
   void Setup(const Shape &in_sample, const LayerConf &conf) override;
   const Shape GetOutputSampleShape() const override {
     CHECK(out_sample_shape_.size()) << "You may haven't call Setup()";
     return out_sample_shape_;
   }
   /// Slice all tensors from 'inputs'
   /// Return a vector including the result of the summation
   const vector<Tensor> Forward(int flag,
                                const vector<Tensor> &inputs) override;

   /// 'grads' should include only one tensor
   /// the first result vector includes the gradients for each input layer
   /// the second result vector is empty
   const std::pair<vector<Tensor>, vector<Tensor> >
   Backward(int flag, const vector<Tensor> &grads) override;

 protected:
  Shape out_sample_shape_;
  size_t axis_ = 1;
  size_t input_size_ = 3;
};
}  // namespace singa
#endif  // SRC_MODEL_LAYER_CONCAT_H_
