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

#ifndef SINGA_MODEL_LOSS_H_
#define SINGA_MODEL_LOSS_H_
#include <stack>
#include "singa/proto/model.pb.h"
#include "singa/core/tensor.h"
namespace singa {

/// The base loss class, which declares the APIs for computing the objective
/// score (loss) for a pair of prediction (from the model) and the target (i.e.
/// the ground truth). It also computes the gradients of the objective w.r.t.
/// the prediction. It has similar APIs as Layer.
// template <typename T = Tensor>
class Loss {
public:
  Loss() = default;
  void Setup(const string &conf) {
    LossConf loss;
    loss.ParseFromString(conf);
    Setup(loss);
  }
  virtual ~Loss() {};
  virtual void ToDevice(std::shared_ptr<Device> device) {}
  /// Set meta fields from user configurations.
  virtual void Setup(const LossConf &conf) {}
  virtual void Setup(const LayerConf &conf) {}

  /// Compute the loss values for each sample/instance given the prediction
  /// and the target.
  virtual Tensor Forward(int flag, const Tensor &prediction,
                         const Tensor &target) = 0;
  virtual Tensor Forward(int flag, const vector<Tensor>& inputs) = 0;

  /// Average loss values for all samples in the mini-batch
  /// It calls Forward() internally. The calling pattern should be
  /// [Evaluate|Forward] Backward.
  float Evaluate(int flag, const Tensor &prediction, const Tensor &target) {
    Tensor loss = Forward(flag, prediction, target);
    return Sum<float>(loss) / (1.0f * loss.Size());
  }

  /// Tensor inputs contains three set of samples, i.e.,
  /// anchor, positive and negative.
  virtual float Evaluate(int flag, const vector<Tensor> &inputs) = 0;

  /// Compute the gradients of the loss values w.r.t. the prediction.
  virtual Tensor Backward() = 0;
  virtual vector<Tensor> Backward(int flag, const Tensor& grads) = 0;
};

// ============= Mean Squared Error ===========================================
/// MSE is for mean squared error or squared euclidean distance.
class MSE : public Loss {
 public:
  /// Compute the loss values for each sample/instance given the prediction
  /// and the target, which is 0.5/||prediction-target||^2
  /// Users can call Average(const Tensor&) to get the average
  /// loss value over all samples in the batch.
  Tensor Forward(int flag, const Tensor& prediction, const Tensor& target) override;
  Tensor Forward(int flag, const vector<Tensor>& inputs) override {
    Tensor out;
    return out;
  }

  /// Compute the gradients of the loss values w.r.t. the prediction,
  /// which is (prediction-target)/batchsize
  Tensor Backward() override;
  vector<Tensor> Backward(int flag, const Tensor& grads) {
    vector<Tensor> out;
    return out;
  }

  float Evaluate(int flag, const vector<Tensor> &inputs) override { return 0.f; }

 private:
  // to buffer intermediate data, i.e., prediction-target
  std::stack<Tensor> buf_;
};


// ===============Softamx Cross Entropy =======================================
/// Softmax + cross entropy for multi-category classification
class SoftmaxCrossEntropy : public Loss {
 public:
  /// Compute the loss values for each sample/instance given the prediction
  /// and the target, which is -log(p[idx_truth]), idx_truth is the truth
  /// category's index and p[] is the probability for each category, computed
  /// from Softmax(prediction).
  /// Users can call Average(const Tensor&) to get the average
  /// loss value over all samples in the batch.
  Tensor Forward(int flag, const Tensor& prediction, const Tensor& target) override;
  Tensor Forward(int flag, const vector<Tensor>& inputs) override {
    Tensor out;
    return out;
  }

  /// Compute the gradients of the loss values w.r.t. the prediction,
  /// which is: p[idx] - 1 if idx is the truth category's index; else,
  /// p[idx]
  Tensor Backward() override;
  vector<Tensor> Backward(int flag, const Tensor& grads) override {
    vector<Tensor> out;
    return out;
  }

  float Evaluate(int flag, const vector<Tensor> &inputs) override { return 0.f; }

 private:
  // to buffer intermediate data, i.e., probability for each category and
  // the target (ground truth)
  std::stack<Tensor> buf_;
};

// ===============Triplet Error =======================================
/// Tripletloss for metric learning
class TripletLoss : public Loss {
 public:
  void Setup(const LayerConf &conf) override;
  Tensor Forward(int flag, const Tensor& prediction, const Tensor& target) {
    return prediction;
  }

  Tensor Backward() {
    Tensor out;
    return out;
  }

  /// Compute the loss values for each sample/instance given the inputs,
  /// which is max(0, margin + (p[0] - p[1]) ^ 2 - (p[0] - p[2]) ^ 2), margin is
  /// to distinguish between positive pair and negative pair, p[] is the feature vector
  /// for each sample/instance.
  Tensor Forward(int flag, const vector<Tensor>& inputs) override;

  /// Compute the gradiens of the loss values, return values contain gradients
  /// for anchors, positive samples and negative samples, respectively.
  vector<Tensor> Backward(int flag, const Tensor& grads) override;

  float Evaluate(int flag, const vector<Tensor> &inputs) override {
    Tensor loss = Forward(flag, inputs);
    return Sum<float>(loss) / (1.0f * loss.Size());
  }

  const float margin() { return margin_; }
  const Tensor mask() { return buf_.top(); }

 private:
  // to buffer intermediate data, i.e., probability for each category and
  // the target (ground truth)
  std::stack<Tensor> buf_;
  float margin_ = 1.f;
};

}  // namespace singa

#endif  // singa_model_loss_h_
