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

#include "singa/model/feed_forward_net.h"
#include "singa/model/initializer.h"
#include "singa/utils/logging.h"
#include "singa/utils/channel.h"
#include <unordered_map>
#include <queue>
namespace singa {

FeedForwardNet::~FeedForwardNet() {
}

static bool IsLayerExistent(std::shared_ptr<Layer> layer, const vector<std::shared_ptr<Layer>> layers) {
  for (size_t i = 0; i < layers.size(); i++)
    if (layers.at(i)->name() == layer->name()) return true;
  return false;
}

std::shared_ptr<Layer> FeedForwardNet::Add(std::shared_ptr<Layer> layer) {
  if (layers_.size() > 0) {
    size_t size = layers_.size();
    dst_[layers_.at(size-1)].push_back(layer);
    src_[layer].push_back(layers_.at(size-1));
    layer_type_[layer] = "onetoone";
  }
  layers_.push_back(layer);
  return layer;
}

std::shared_ptr<Layer> FeedForwardNet::Add(const LayerConf& conf,
    const Shape* sample_shape) {
  std::shared_ptr<Layer> layer(CreateLayer(conf.type()));
  CHECK(conf.has_name()) << "Must set layer name";
  if (sample_shape == nullptr)
    layer->Setup(layers_.back()->GetOutputSampleShape(), conf);
  else
    layer->Setup(*sample_shape, conf);
  if (layer->layer_type() == "Split" || layer->layer_type() == "Slice")
    layer_type_[layer] = "onetomany";
  else layer_type_[layer] = "onetoone";
  Add(layer);
  LOG(INFO) << layer->name() << VecToStr(layer->GetOutputSampleShape());
  return layer;
}

std::shared_ptr<Layer> FeedForwardNet::Add(const LayerConf& conf,
              std::shared_ptr<Layer> src) {
  std::shared_ptr<Layer> layer(CreateLayer(conf.type()));
  /// src should not be nullptr
  if (src == nullptr)
    LOG(FATAL) << "src cannot be nullptr.";
  else
    layer->Setup(src->GetOutputSampleShape(), conf);
  if (src_.find(layer) != src_.end())
    LOG(FATAL) << "layer " << layer->name() << " already exists.";
  if (!IsLayerExistent(src, layers_))
    LOG(FATAL) << "src layer " << src->name() << " does not exist.";
  src_[layer].push_back(src);
  layer_type_[layer] = "onetoone";
  if (dst_.find(src) != dst_.end()) {
    dst_[src].push_back(layer);
    if (dst_[src].size() > 1) { /// multiple dst layers, onetomany
      if (layer_type_[src] == "manytoone")
        LOG(FATAL) << "network does not support many-to-many layer";
      layer_type_[src] = "onetomany";
    }
  } else {
    dst_[src].push_back(layer);
    if (layer_type_[src] != "manytoone")
      layer_type_[src] = "onetoone";
  }
  layers_.push_back(layer);
  LOG(INFO) << layer->name() << VecToStr(layer->GetOutputSampleShape());
  return layer;
}

std::shared_ptr<Layer> FeedForwardNet::Add(const LayerConf& conf,
         vector<std::shared_ptr<Layer>> srcs) {
  std::shared_ptr<Layer> layer(CreateLayer(conf.type()));
  /// srcs should not be nullptr
  if (!srcs.size() || (srcs.size() == 1 && srcs.at(0) == nullptr))
    LOG(FATAL) << "src cannot be nullptr.";
  else {
    if (layer->layer_type() == "Merge" || layer->layer_type() == "Concat")
      layer->Setup(srcs.at(0)->GetOutputSampleShape(), conf);
    else
      LOG(FATAL) << "layer " << layer->name()
            << " should not have multiple src layers";
  }
  if (src_.find(layer) != src_.end())
   LOG(FATAL) << "layer " << layer->name() << " already exists.";
  vector<std::shared_ptr<Layer>> tmpsrc;
  for (size_t i = 0; i < srcs.size(); i++) {
    if (!IsLayerExistent(srcs.at(i), layers_))
      LOG(FATAL) << "src layer " << srcs.at(i)->name() << " does not exist.";
    tmpsrc.push_back(srcs.at(i));
  }
  src_[layer] = tmpsrc;
  if (srcs.size() == 1) layer_type_[layer] = "onetoone";
  else layer_type_[layer] = "manytoone";

  for (size_t i = 0; i < srcs.size(); i++) {
    std::shared_ptr<Layer> src = srcs.at(i);
    if (dst_.find(src) != dst_.end()) {
      dst_[src].push_back(layer);
      if (dst_[src].size() > 1) { /// multiple dst layers, onetomany
        if (layer_type_[src] == "manytoone")
          LOG(FATAL) << "network does not support many-to-many layer";
        layer_type_[src] = "onetomany";
      }
    } else {
      dst_[src].push_back(layer);
      if (layer_type_[src] != "manytoone")
        layer_type_[src] = "onetoone";
    }
  }
  layers_.push_back(layer);
  LOG(INFO) << layer->name() << VecToStr(layer->GetOutputSampleShape());
  return layer;
}

const vector<string> FeedForwardNet::GetParamNames() const {
  vector<string> names;
  for (auto layer : layers_)
    for (const auto name : layer->param_names()) names.push_back(name);
  return names;
}
const vector<Tensor> FeedForwardNet::GetParamValues() const {
  vector<Tensor> values;
  for (auto layer : layers_)
    for (const auto value : layer->param_values()) values.push_back(value);
  return values;
}

const vector<ParamSpec> FeedForwardNet::GetParamSpecs() const {
  vector<ParamSpec> specs;
  for (auto layer : layers_)
    for (const auto spec : layer->param_specs()) specs.push_back(spec);
  return specs;
}

void FeedForwardNet::Compile(bool shuffle, Optimizer* opt, Loss* loss,
                             Metric* metric) {
  std::shared_ptr<Updater> updater = std::make_shared<Updater>(opt);
  Compile(shuffle, true, updater, loss, metric);
}

void FeedForwardNet::Compile(bool shuffle, bool to_register,
          std::shared_ptr<Updater> updater, Loss* loss, Metric* metric) {
  shuffle_ = shuffle;
  bool train = (updater != nullptr) && (loss != nullptr);
  bool test = metric != nullptr;
  CHECK(train || test) << "Must set updater and loss, or set metric";
  updater_ = updater;
  loss_ = loss;
  metric_ = metric;
  const auto specs = GetParamSpecs();
  auto params = GetParamValues();
  CHECK_EQ(specs.size(), params.size());
  for (size_t k = 0; k < specs.size(); k++) {
    if (to_register) {
      updater_->Register(specs[k].name(), specs[k]);
    }
    auto init = CreateInitializer(specs[k].filler());
    init->Fill(params[k]);
    LOG(INFO) << specs[k].name() << " : " << params[k].L1();
  }
}

void FeedForwardNet::ToDevice(std::shared_ptr<Device> device) {
  for (auto layer : layers_) layer->ToDevice(device);
  /*
  opt_->ToDevice(device);
  loss_->ToDevice(device);
  metric_->ToDevice(device);
  */
}

FeedForwardNet FeedForwardNet::Clone(std::shared_ptr<Device> device) {
  FeedForwardNet net;
  /*
  for (auto layer: layers_)
    net.layers_.push_back(layer->CloneTo(device));
  if (opt_ != nullptr)
    net.opt_ = opt_->CloneTo(device);
  if (loss_ != nullptr)
    net.loss_ = loss_.CloneTo(device);
  if (metric_ != nullptr)
    net.metric_ = metric_->CloneTo(device);
  net.shuffle_ = shuffle_;
  net.device_ = device;
  net.dtype_ = dtype;
  */
  return net;
}

void FeedForwardNet::AsType(DataType dtype) {
  LOG(FATAL) << "FeedForwardNet::AsType not implemented";
}

void FeedForwardNet::Train(int nb_epoch, const Tensor& x, const Tensor& y,
                           size_t batchsize, float val_split) {
  CHECK_EQ(x.shape(0), y.shape(0)) << "Diff num of sampels in x and y";
  size_t num_train = x.shape(0) * val_split;
  if (val_split == 0.0f) {
    Tensor dummy;
    Train(nb_epoch, x, y, batchsize, dummy, dummy);
  } else {
    const Tensor train_x = CopyRows(x, 0, num_train);
    const Tensor train_y = CopyRows(y, 0, num_train);
    const Tensor test_x = CopyRows(x, num_train, x.shape(0));
    const Tensor test_y = CopyRows(y, num_train, y.shape(0));
    Train(nb_epoch, train_x, train_y, batchsize, test_x, test_y);
  }
}

void FeedForwardNet::Train(int nb_epoch, const Tensor& x, const Tensor& y,
                           size_t batchsize, const Tensor& val_x,
                           const Tensor& val_y) {
  CHECK_EQ(x.shape(0), y.shape(0)) << "Diff num of sampels in x and y";
  int num_extra_samples = x.shape(0) % batchsize;
  if (num_extra_samples != 0)
    LOG(WARNING) << "Pls set batchsize to make num_total_samples "
                 << "% batchsize == 0. Otherwise, the last "
                 << num_extra_samples << " samples would not be used";
  Channel* train_ch = GetChannel("train_perf");
  train_ch->EnableDestStderr(true);
  Channel* val_ch = GetChannel("val_perf");
  val_ch->EnableDestStderr(true);
  std::vector<size_t> index;
  for (size_t i = 0; i < x.shape(0) / batchsize; i++) index.push_back(i);
  for (int epoch = 0; epoch < nb_epoch; epoch++) {
    if (shuffle_) std::random_shuffle(index.begin(), index.end());
    float loss = 0.0f, metric = 0.0f;
    size_t b = 0;
    for (; b < x.shape(0) / batchsize; b++) {
      size_t idx = index[b];
      const Tensor bx = CopyRows(x, idx * batchsize, (idx + 1) * batchsize);
      const Tensor by = CopyRows(y, idx * batchsize, (idx + 1) * batchsize);
      const auto ret = TrainOnBatch(epoch, bx, by);
      loss += ret.first;
      metric += ret.second;
    }
    if (val_x.Size() == 0) continue;
    loss /= b;
    metric /= b;
    train_ch->Send(
        "Epoch " + std::to_string(epoch) + ", training loss = " +
        std::to_string(loss) + ", accuracy = " + std::to_string(metric) +
        ", lr = " +
        std::to_string(updater_->GetOptimizer()->GetLearningRate(epoch)));
    if (val_x.Size() && val_y.Size()) {
      const auto val_perf = Evaluate(val_x, val_y, batchsize);
      val_ch->Send("Epoch " + std::to_string(epoch) + ", val loss = " +
                   std::to_string(Sum(val_perf.first) / val_y.Size()) +
                   ", metric = " +
                   std::to_string(Sum(val_perf.second) / val_y.Size()));
    }
  }
}

void FeedForwardNet::Train(int nb_epoch, const Tensor& x, size_t batchsize,
                           float val_split) {
  size_t num_train = x.shape(0) * val_split;
  if (val_split == 0.0f) {
    Tensor dummy;
    Train(nb_epoch, x, batchsize, dummy);
  } else {
    const Tensor train_x = CopyRows(x, 0, num_train);
    const Tensor test_x = CopyRows(x, num_train, x.shape(0));
    Train(nb_epoch, train_x, batchsize, test_x);
  }
}

void FeedForwardNet::Train(int nb_epoch, const Tensor& x, size_t batchsize,
                           const Tensor& val_x) {
  int num_extra_samples = x.shape(0) % batchsize;
  if (num_extra_samples != 0)
    LOG(WARNING) << "Pls set batchsize to make num_total_samples "
                 << "% batchsize == 0. Otherwise, the last "
                 << num_extra_samples << " samples would not be used";
  Channel* train_ch = GetChannel("train_perf");
  train_ch->EnableDestStderr(true);
  Channel* val_ch = GetChannel("val_perf");
  val_ch->EnableDestStderr(true);
  std::vector<size_t> index;
  for (size_t i = 0; i < x.shape(0) / batchsize; i++) index.push_back(i);
  for (int epoch = 0; epoch < nb_epoch; epoch++) {
    if (shuffle_) std::random_shuffle(index.begin(), index.end());
    float loss = 0.0f;
    size_t b = 0;
    for (; b < x.shape(0) / batchsize; b++) {
      size_t idx = index[b];
      const Tensor bx = CopyRows(x, idx * batchsize, (idx + 1) * batchsize);
      const auto ret = TrainOnBatch(epoch, bx);
      loss += ret.first;
    }
    if (val_x.Size() == 0) continue;
    loss /= b;
    train_ch->Send(
        "Epoch " + std::to_string(epoch) + ", training loss = " +
        std::to_string(loss) + ", lr = " +
        std::to_string(updater_->GetOptimizer()->GetLearningRate(epoch)));
    if (val_x.Size()) {
      const auto val_perf = Evaluate(val_x, batchsize);
      val_ch->Send("Epoch " + std::to_string(epoch) + ", val loss = " +
                   std::to_string(Sum(val_perf.first) / val_x.shape(0)));
    }
  }
}

const std::pair<float, float> FeedForwardNet::TrainOnBatch(int epoch,
                                                           const Tensor& x,
                                                           const Tensor& y) {
  int flag = kTrain;
  const Tensor fea = Forward(flag, x);
  float loss = loss_->Evaluate(flag, fea, y);
  float metric = metric_->Evaluate(fea, y);
  const Tensor grad = loss_->Backward();
  auto grads = Backward(kTrain, grad / static_cast<float>(x.shape(0)));
  auto names = GetParamNames();
  auto values = GetParamValues();
  for (size_t k = 0; k < grads.size(); k++) {
    updater_->Apply(epoch, names[k], grads[k], values.at(k));
  }
  return std::make_pair(loss, metric);
}

const std::pair<float, float> FeedForwardNet::TrainOnBatch(int epoch,
                                                           const Tensor& x) {
  int flag = kTrain;
  Tensor fea = Forward(flag, x);
  size_t input_size = 3;
  Shape shape = fea.shape();
  // LOG(INFO) << "output shape: " << VecToStr(shape); //
  auto dev = fea.device();
  fea.ToHost();
  shape.at(0) /= input_size;
  size_t size = fea.Size() / input_size;
  const float* data = fea.data<float>();
  vector<Tensor> feat;
  for (size_t i = 0; i < input_size; i++) {
    Tensor tmp(shape);
    tmp.CopyDataFromHostPtr(data + i * size, size);
    //tmp.ToDevice(dev);
    feat.push_back(tmp);
  }
  //LOG(INFO) << "compute loss...";
  float loss = loss_->Evaluate(flag, feat);
  LOG(INFO) << "loss is : " << loss << ", size: " << size;
  vector<Tensor> grad = loss_->Backward(flag, x);
  Tensor out_grad(fea.shape());
  for (size_t i = 0; i < grad.size(); i++) {
    Tensor tmp = grad.at(i);
    tmp.ToHost();
    tmp *= 3 / static_cast<float>(x.shape(0));
    const float* d = tmp.data<float>();
    out_grad.CopyDataFromHostPtr(d, size, i * size);
  }
  out_grad.ToDevice(dev);
  //LOG(INFO) << "get out grad shape: " << VecToStr(out_grad.shape());
  auto grads = Backward(kTrain, out_grad);
  //LOG(INFO) << "get dev of parameters";
  auto names = GetParamNames();
  auto values = GetParamValues();
  for (size_t k = 0; k < grads.size(); k++) {
    updater_->Apply(epoch, names[k], grads[k], values.at(k));
  }
  return std::make_pair(loss, 0.f);
}

const Tensor FeedForwardNet::Forward(int flag, const Tensor& data) {
  /*Tensor input = data, output;
  // LOG(INFO) << data.L1();
  for (auto layer : layers_) {
    output = layer->Forward(flag, input);
    // LOG(INFO) << layer->name() << ": " << output.L2();
    input = output;
  }
  return output;*/

  unordered_map<std::shared_ptr<Layer>, std::queue<Tensor>> output;
  output[nullptr].push(data);

  /// layers_ are topological sorted
  for (auto layer : layers_) {
     /* LOG(INFO) << "forward: " << layer->name(); //
     string names = "";
     if (src_.find(layer) != src_.end())
       for (size_t i = 0; i < src_[layer].size(); i++)
         names += src_[layer].at(i)->name() + ",";
     LOG(INFO) << "srcs: " << names;
     names = "";
     if (dst_.find(layer) != dst_.end())
       for (size_t i = 0; i < dst_[layer].size(); i++)
         names += dst_[layer].at(i)->name() + ",";
       LOG(INFO) << "dsts: " << names;*/
    if (layer_type_[layer] == "onetomany") {
      vector<Tensor> tmp;
      std::shared_ptr<Layer> tmplayer = nullptr;
      if (src_.find(layer) != src_.end()) tmplayer = src_[layer].at(0);
      tmp.push_back(output[tmplayer].front());
      // LOG(INFO) << "input shape: " << VecToStr(output[tmplayer].front().shape()); //
      output[tmplayer].pop();
      if (output[tmplayer].empty()) output.erase(tmplayer);
      // LOG(INFO) << tmp.size() << " , " << VecToStr(tmp.at(0).shape()); //
      for (auto x : layer->Forward(flag, tmp))
        output[layer].push(x);
    } else if (layer_type_[layer] == "manytoone") {
      vector<Tensor> tmp;
      for (size_t i = 0; i < src_[layer].size(); i++) {
        std::shared_ptr<Layer> tmplayer = src_[layer].at(i);
        tmp.push_back(output[tmplayer].front());
        // Shape shape = output[tmplayer].front().shape(); //
        // LOG(INFO) << "input shape: " << VecToStr(shape); //
        output[tmplayer].pop();
        if (output[tmplayer].empty()) output.erase(tmplayer);
      }
      output[layer].push(layer->Forward(flag, tmp).at(0));
    } else if (layer_type_[layer] == "onetoone") {
      std::shared_ptr<Layer> tmplayer = nullptr;
      if (src_.find(layer) != src_.end()) // not start layer
        tmplayer = src_[layer].at(0);
      output[layer].push(layer->Forward(flag, output[tmplayer].front()));
      // LOG(INFO) << "input shape: " << VecToStr(output[tmplayer].front().shape()); //
      output[tmplayer].pop();
      if (output[tmplayer].empty()) output.erase(tmplayer);

    } else
      LOG(FATAL) << "invalid layer type: " << layer_type_[layer];
    // LOG(INFO) << "==================================="; //
  }

  Tensor out = output[layers_.back()].front();
  output.erase(layers_.back());
  return out;
}

const Tensor FeedForwardNet::Forward(int flag, const Tensor& data,
    const string layer_name) {
  unordered_map<std::shared_ptr<Layer>, std::queue<Tensor>> output;
  output[nullptr].push(data);

  /// layers_ are topological sorted
  for (auto layer : layers_) {
     LOG(INFO) << "forward: " << layer->name(); //
    if (layer_type_[layer] == "onetomany") {
      vector<Tensor> tmp;
      std::shared_ptr<Layer> tmplayer = nullptr;
      if (src_.find(layer) != src_.end()) tmplayer = src_[layer].at(0);
      tmp.push_back(output[tmplayer].front());
      // LOG(INFO) << "input shape: " << VecToStr(output[tmplayer].front().shape()); //
      output[tmplayer].pop();
      if (output[tmplayer].empty()) output.erase(tmplayer);

      for (auto x : layer->Forward(flag, tmp))
        output[layer].push(x);
    } else if (layer_type_[layer] == "manytoone") {
      vector<Tensor> tmp;
      for (size_t i = 0; i < src_[layer].size(); i++) {
        std::shared_ptr<Layer> tmplayer = src_[layer].at(i);
        tmp.push_back(output[tmplayer].front());
        // Shape shape = output[tmplayer].front().shape(); //
        // LOG(INFO) << "input shape: " << VecToStr(shape); //
        output[tmplayer].pop();
        if (output[tmplayer].empty()) output.erase(tmplayer);
      }
      output[layer].push(layer->Forward(flag, tmp).at(0));
    } else if (layer_type_[layer] == "onetoone") {
      std::shared_ptr<Layer> tmplayer = nullptr;
      if (src_.find(layer) != src_.end()) // not start layer
        tmplayer = src_[layer].at(0);
      output[layer].push(layer->Forward(flag, output[tmplayer].front()));
      // LOG(INFO) << "input shape: " << VecToStr(output[tmplayer].front().shape()); //
      output[tmplayer].pop();
      if (output[tmplayer].empty()) output.erase(tmplayer);
    } else
      LOG(FATAL) << "invalid layer type: " << layer_type_[layer];
    // LOG(INFO) << "==================================="; //
    LOG(INFO) << "layer_name: " << layer_name;
    if (layer_name == layer->name()) {
      Tensor out = output[layer].front();
      output.erase(layer);
      return out;
    }
  }

  Tensor out = output[layers_.back()].front();
  output.erase(layers_.back());
  return out;
}

const vector<Tensor> FeedForwardNet::Backward(int flag, const Tensor& grad) {
  /*vector<Tensor> param_grads;
  std::stack<Tensor> buf;
  Tensor tmp = grad;
  for (int i = layers_.size() - 1; i >= 0; i--) {
    // LOG(INFO) << layers_.at(i)->name() << " : " << tmp.L1();
    auto ret = layers_.at(i)->Backward(flag, tmp);
    tmp = ret.first;
    if (ret.second.size()) {
      for (int k = ret.second.size() - 1; k >= 0; k--) {
        buf.push(ret.second[k]);
        // LOG(INFO) <<  "      " << buf.top().L1();
      }
    }
  }*/

  vector<Tensor> param_grads;
  std::stack<Tensor> buf;
  unordered_map<std::shared_ptr<Layer>, std::stack<Tensor>> output_grads;
  output_grads[nullptr].push(grad);
  for (int i = layers_.size() - 1; i >= 0; i--) {
    std::shared_ptr<Layer> layer = layers_.at(i);
    // LOG(INFO) << "backward: " << layer->name(); //
    if (layer_type_[layer] == "onetoone") {
      std::shared_ptr<Layer> tmplayer = nullptr;
      if (dst_.find(layer) != dst_.end()) // not last layer
        tmplayer = dst_[layer].at(0);
      // LOG(INFO) << "top grads shape: " << VecToStr(output_grads[tmplayer].top().shape()); //
      auto ret = layer->Backward(flag, output_grads[tmplayer].top());
      output_grads[tmplayer].pop();
      if (output_grads[tmplayer].empty()) output_grads.erase(tmplayer);

      output_grads[layer].push(ret.first);
      if (ret.second.size()) {
        for (int j = ret.second.size() - 1; j >= 0; j--)
          buf.push(ret.second[j]);
      }
    } else if (layer_type_[layer] == "onetomany") {
      vector<Tensor> tmp;
      if (dst_.find(layer) == dst_.end()) { // TODO(jixin)
        LOG(FATAL) << "current model does not support multiple outputs";
      } else {
        //for (int j = dst_[layer].size() - 1; j >= 0; j--) {
        for (size_t j = 0; j < dst_[layer].size(); j++) {
          std::shared_ptr<Layer> tmplayer = dst_[layer].at(j);
          // LOG(INFO) << "top grads shape: " << VecToStr(output_grads[tmplayer].top().shape()); //
          tmp.push_back(output_grads[tmplayer].top());
          output_grads[tmplayer].pop();
          if (output_grads[tmplayer].empty()) output_grads.erase(tmplayer);
        }
      }
      auto ret = layer->Backward(flag, tmp);
      output_grads[layer].push(ret.first[0]);
      if (ret.second.size())
        LOG(FATAL) << "split and slice layer should not have parameters.";
    } else if (layer_type_[layer] == "manytoone") {
      std::shared_ptr<Layer> tmplayer = nullptr;
      if (dst_.find(layer) != dst_.end()) // not last layer
        tmplayer = dst_[layer].at(0);
      // LOG(INFO) << "top grads shape: " << VecToStr(output_grads[tmplayer].top().shape()); //
      auto ret = layer->Backward(flag, vector<Tensor>{output_grads[tmplayer].top()});
      output_grads[tmplayer].pop();
      if (output_grads[tmplayer].empty()) output_grads.erase(tmplayer);

      for (int j = ret.first.size() - 1; j >= 0; j--)
        output_grads[layer].push(ret.first[j]);
      if (ret.second.size())
        LOG(FATAL) << "merge and concatenate layer should not have parameters.";
    }
    //  LOG(INFO) << "==================================="; //
  }

  while (!buf.empty()) {
    param_grads.push_back(buf.top());
    buf.pop();
  }
  return param_grads;
}

Tensor FeedForwardNet::Extract(const Tensor& x, const string layer_name) {
  int flag = kEval;
  return FeedForwardNet::Forward(flag, x, layer_name);
}

std::pair<Tensor, Tensor> FeedForwardNet::Evaluate(const Tensor& x,
                                                   size_t batchsize) {
  CHECK_GE(x.shape(0), batchsize);
  int num_extra_samples = x.shape(0) % batchsize;
  Tensor loss(Shape{x.shape(0)}), metric(Shape{x.shape(0)});
  for (size_t b = 0; b < x.shape(0) / batchsize; b++) {
    int start = b * batchsize, end = start + batchsize;
    const Tensor bx = CopyRows(x, start, end);
    const auto ret = EvaluateOnBatch(bx);
    CopyDataToFrom(&loss, ret.first, batchsize, start, 0);
  }
  {
    int start = x.shape(0) - batchsize, end = x.shape(0);
    const Tensor bx = CopyRows(x, start, end);
    const auto ret = EvaluateOnBatch(bx);
    int dst_offset = x.shape(0) - num_extra_samples;
    int src_offset = batchsize - num_extra_samples;
    CopyDataToFrom(&loss, ret.first, num_extra_samples, dst_offset, src_offset);
  }
  return std::make_pair(loss, metric);
}

std::pair<Tensor, Tensor> FeedForwardNet::EvaluateOnBatch(const Tensor& x) {
  int flag = kEval;
  const Tensor fea = Forward(flag, x);
  size_t input_size = 3;
  Shape shape = fea.shape();
  shape.at(0) /= 3;
  size_t size = fea.Size() / input_size;
  vector<Tensor> feat;
  const float *data = fea.data<float>();
  for (size_t i = 0; i < input_size; i++) {
    Tensor tmp(Shape{shape});
    tmp.CopyDataFromHostPtr(data + i * size, size);
    feat.push_back(tmp);
  }
  const Tensor l = loss_->Forward(flag, feat);
  const Tensor m;
  return std::make_pair(l, m);
}

std::pair<Tensor, Tensor> FeedForwardNet::Evaluate(const Tensor& x,
                                                   const Tensor& y,
                                                   size_t batchsize) {
  CHECK_EQ(x.shape(0), y.shape(0)) << "Diff num of sampels in x and y";
  CHECK_GE(x.shape(0), batchsize);
  int num_extra_samples = x.shape(0) % batchsize;
  Tensor loss(Shape{x.shape(0)}), metric(Shape{x.shape(0)});
  for (size_t b = 0; b < x.shape(0) / batchsize; b++) {
    int start = b * batchsize, end = start + batchsize;
    const Tensor bx = CopyRows(x, start, end);
    const Tensor by = CopyRows(y, start, end);
    const auto ret = EvaluateOnBatch(bx, by);
    CopyDataToFrom(&loss, ret.first, batchsize, start, 0);
    CopyDataToFrom(&metric, ret.second, batchsize, start, 0);
  }
  {
    int start = x.shape(0) - batchsize, end = x.shape(0);
    const Tensor bx = CopyRows(x, start, end);
    const Tensor by = CopyRows(y, start, end);
    const auto ret = EvaluateOnBatch(bx, by);
    int dst_offset = x.shape(0) - num_extra_samples;
    int src_offset = batchsize - num_extra_samples;
    CopyDataToFrom(&loss, ret.first, num_extra_samples, dst_offset, src_offset);
    CopyDataToFrom(&metric, ret.second, num_extra_samples, dst_offset,
                   src_offset);
  }
  return std::make_pair(loss, metric);
}

std::pair<Tensor, Tensor> FeedForwardNet::EvaluateOnBatch(const Tensor& x,
                                                          const Tensor& y) {
  int flag = kEval;
  const Tensor fea = Forward(flag, x);
  const Tensor l = loss_->Forward(flag, fea, y);
  const Tensor m = metric_->Forward(fea, y);
  return std::make_pair(l, m);
}

const Tensor FeedForwardNet::Predict(const Tensor& x, size_t batchsize) {
  CHECK_GE(x.shape(0), batchsize);
  int num_extra_samples = x.shape(0) % batchsize;
  const auto outshape = layers_.back()->GetOutputSampleShape();
  Tensor y(Shape{x.shape(0), Product(outshape)}, x.device());
  for (size_t b = 0; b < x.shape(0) / batchsize; b++) {
    int start = b * batchsize, end = start + batchsize;
    const Tensor bx = CopyRows(x, start, end);
    CopyDataToFrom(&y, PredictOnBatch(bx), batchsize * y.shape(1),
                   start * y.shape(1), 0);
  }
  if (num_extra_samples > 0) {
    int start = x.shape(0) - batchsize, end = x.shape(0);
    const Tensor bx = CopyRows(x, start, end);
    CopyDataToFrom(&y, PredictOnBatch(bx), num_extra_samples * y.shape(1),
                   (x.shape(0) - num_extra_samples) * y.shape(1),
                   (batchsize - num_extra_samples) * y.shape(1));
  }
  return y;
}

const Tensor FeedForwardNet::PredictOnBatch(const Tensor& x) {
  return Forward(kEval, x);
}
}  // namespace singa
