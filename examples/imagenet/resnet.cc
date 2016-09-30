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

#include "singa/singa_config.h"
#ifdef USE_OPENCV
#include <cmath>
#include "./ilsvrc12.h"
#include "singa/io/snapshot.h"
#include "singa/model/feed_forward_net.h"
#include "singa/model/initializer.h"
#include "singa/model/metric.h"
#include "singa/model/optimizer.h"
#include "singa/utils/channel.h"
#include "singa/utils/string.h"
#include "singa/utils/timer.h"
namespace singa {

// currently supports 'cudnn' and 'singacpp'
const std::string engine = "cudnn";
LayerConf GenConvConf(string name, int nb_filter, int kernel, int stride,
                      int pad, float std, float bias = .0f) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type(engine + "_convolution");
  ConvolutionConf *conv = conf.mutable_convolution_conf();
  conv->set_num_output(nb_filter);
  conv->add_kernel_size(kernel);
  conv->add_stride(stride);
  conv->add_pad(pad);
  conv->set_bias_term(true);

  ParamSpec *wspec = conf.add_param();
  wspec->set_name(name + "_weight");
  auto wfill = wspec->mutable_filler();
  wfill->set_type("Gaussian");
  wfill->set_std(std);

  ParamSpec *bspec = conf.add_param();
  bspec->set_name(name + "_bias");
  bspec->set_lr_mult(2);
  bspec->set_decay_mult(0);
  auto bfill = bspec->mutable_filler();
  bfill->set_value(bias);
  return conf;
}

LayerConf GenPoolingConf(string name, bool max_pool, int kernel, int stride,
                         int pad) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type(engine + "_pooling");
  PoolingConf *pool = conf.mutable_pooling_conf();
  pool->set_kernel_size(kernel);
  pool->set_stride(stride);
  pool->set_pad(pad);
  if (!max_pool) pool->set_pool(PoolingConf_PoolMethod_AVE);
  return conf;
}

LayerConf GenReLUConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type(engine + "_relu");
  return conf;
}

LayerConf GenDenseConf(string name, int num_output, float std, float wd,
                       float bias = .0f) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("singa_dense");
  DenseConf *dense = conf.mutable_dense_conf();
  dense->set_num_output(num_output);

  ParamSpec *wspec = conf.add_param();
  wspec->set_name(name + "_weight");
  wspec->set_decay_mult(wd);
  auto wfill = wspec->mutable_filler();
  wfill->set_type("Gaussian");
  wfill->set_std(std);

  ParamSpec *bspec = conf.add_param();
  bspec->set_name(name + "_bias");
  bspec->set_lr_mult(2);
  bspec->set_decay_mult(0);
  auto bfill = bspec->mutable_filler();
  bfill->set_value(bias);

  return conf;
}

LayerConf GenBatchNormConf(string name, float std) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type(engine + "_batchnorm");
  BatchNormConf *bn = conf.mutable_batchnorm_conf();
  bn->set_factor(0.9);

  ParamSpec *wspec = conf.add_param();
  wspec->set_name(name + "_weight");
  auto wfill = wspec->mutable_filler();
  wfill->set_type("Constant");
  wfill->set_value(1);

  ParamSpec *bspec = conf.add_param();
  bspec->set_name(name + "_bias");
  auto bfill = bspec->mutable_filler();
  bfill->set_type("Constant");
  bfill->set_value(0);

  ParamSpec *meanspec = conf.add_param();
  meanspec->set_name(name + "_mean");
  auto meanfill = meanspec->mutable_filler();
  meanfill->set_type("Constant");
  meanfill->set_value(0);

  ParamSpec *varspec = conf.add_param();
  varspec->set_name(name + "_variance");
  auto varfill = varspec->mutable_filler();
  varfill->set_type("Constant");
  varfill->set_value(0);
  return conf;
}

LayerConf GenSplitConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("singa_split");
  return conf;
}

LayerConf GenMergeConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("singa_merge");
  return conf;
}

LayerConf GenSoftmaxConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type(engine + "_softmax");
  return conf;
}

LayerConf GenFlattenConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("singa_flatten");
  return conf;
}

LayerConf GenDropoutConf(string name, float dropout_ratio) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type(engine + "_dropout");
  DropoutConf *dropout = conf.mutable_dropout_conf();
  dropout->set_dropout_ratio(dropout_ratio);
  return conf;
}

std::shared_ptr<Layer> BuildingBlock(FeedForwardNet& net, string layer_name, int nb_filter,
            int stride, float std, std::shared_ptr<Layer> src) {
  std::shared_ptr<Layer> split = net.Add(GenSplitConf("split" + layer_name), src);
  std::shared_ptr<Layer> bn_br1 = nullptr;
  if (stride > 1) {
    net.Add(GenConvConf("conv" + layer_name + "_br1", nb_filter, 1, stride, 0, std), split);
    bn_br1 = net.Add(GenBatchNormConf("bn" + layer_name + "_br1", std));
  }
  net.Add(GenConvConf("conv" + layer_name + "_br2a", nb_filter, 3, stride, 1, std), split);
  net.Add(GenBatchNormConf("bn" + layer_name + "_br2a", std));
  net.Add(GenReLUConf("relu" + layer_name + "_br2a"));
  net.Add(GenConvConf("conv" + layer_name + "_br2b", nb_filter, 3, 1, 1, std));
  std::shared_ptr<Layer> bn2 = net.Add(GenBatchNormConf("bn" + layer_name + "_br2b", std));
  if (stride > 1)
   return net.Add(GenMergeConf("merge" + layer_name), vector<std::shared_ptr<Layer>>{bn_br1, bn2});
  else return net.Add(GenMergeConf("merge" + layer_name), vector<std::shared_ptr<Layer>>{split, bn2});
}


FeedForwardNet CreateNet() {
  FeedForwardNet net;
  Shape s{3, 224, 224};

  net.Add(GenConvConf("conv1", 64, 7, 2, 3, 0.01), &s);
  net.Add(GenBatchNormConf("bn1", 0.01));
  net.Add(GenReLUConf("relu1"));
  std::shared_ptr<Layer> pool1 = net.Add(GenPoolingConf("pool1", true, 3, 2, 1));

  std::shared_ptr<Layer> b2a = BuildingBlock(net, "2a", 64, 1, 0.01, pool1);
  std::shared_ptr<Layer> b2b = BuildingBlock(net, "2b", 64, 1, 0.01, b2a);
  //std::shared_ptr<Layer> b2c = BuildingBlock(net, "2c", 64, 1, 0.01, b2b);

  std::shared_ptr<Layer> b3a = BuildingBlock(net, "3a", 128, 2, 0.01, b2b);
  std::shared_ptr<Layer> b3b = BuildingBlock(net, "3b", 128, 1, 0.01, b3a);
  //std::shared_ptr<Layer> b3c = BuildingBlock(net, "3c", 128, 1, 0.01, b3b);
  //std::shared_ptr<Layer> b3d = BuildingBlock(net, "3d", 128, 1, 0.01, b3c);

  std::shared_ptr<Layer> b4a = BuildingBlock(net, "4a", 256, 2, 0.01, b3b);
  std::shared_ptr<Layer> b4b = BuildingBlock(net, "4b", 256, 1, 0.01, b4a);
  //std::shared_ptr<Layer> b4c = BuildingBlock(net, "4c", 256, 1, 0.01, b4b);
  //std::shared_ptr<Layer> b4d = BuildingBlock(net, "4d", 256, 1, 0.01, b4c);
  //std::shared_ptr<Layer> b4e = BuildingBlock(net, "4e", 256, 1, 0.01, b4d);
  //std::shared_ptr<Layer> b4f = BuildingBlock(net, "4f", 256, 1, 0.01, b4e);

  std::shared_ptr<Layer> b5a = BuildingBlock(net, "5a", 512, 2, 0.01, b4b);
  //std::shared_ptr<Layer> b5b = BuildingBlock(net, "5b", 512, 1, 0.01, b5a);
  BuildingBlock(net, "5b", 512, 1, 0.01, b5a);

  net.Add(GenPoolingConf("pool5", false, 7, 1, 0));
  net.Add(GenFlattenConf("flat"));
  net.Add(GenDenseConf("ip6", 1000, 0.01, 1));

  return net;
}

FeedForwardNet CreateNet2() {
  FeedForwardNet net;
  Shape s{3, 224, 224};

  net.Add(GenConvConf("conv1", 64, 7, 2, 3, 0.01), &s);
  net.Add(GenBatchNormConf("bn1", 0.01));
  net.Add(GenReLUConf("relu1"));
  std::shared_ptr<Layer> pool1 = net.Add(GenPoolingConf("pool1", true, 3, 2, 1));

  std::shared_ptr<Layer> b2a = BuildingBlock(net, "2a", 64, 1, 0.01, pool1);
  std::shared_ptr<Layer> b2b = BuildingBlock(net, "2b", 64, 1, 0.01, b2a);
  //std::shared_ptr<Layer> b2c = BuildingBlock(net, "2c", 64, 1, 0.01, b2b);

  std::shared_ptr<Layer> b3a = BuildingBlock(net, "3a", 128, 2, 0.01, b2b);
  std::shared_ptr<Layer> b3b = BuildingBlock(net, "3b", 128, 1, 0.01, b3a);
  //std::shared_ptr<Layer> b3c = BuildingBlock(net, "3c", 128, 1, 0.01, b3b);
  //std::shared_ptr<Layer> b3d = BuildingBlock(net, "3d", 128, 1, 0.01, b3c);

  std::shared_ptr<Layer> b4a = BuildingBlock(net, "4a", 256, 2, 0.01, b3b);
  std::shared_ptr<Layer> b4b = BuildingBlock(net, "4b", 256, 1, 0.01, b4a);
  //std::shared_ptr<Layer> b4c = BuildingBlock(net, "4c", 256, 1, 0.01, b4b);
  //std::shared_ptr<Layer> b4d = BuildingBlock(net, "4d", 256, 1, 0.01, b4c);
  //std::shared_ptr<Layer> b4e = BuildingBlock(net, "4e", 256, 1, 0.01, b4d);
  //std::shared_ptr<Layer> b4f = BuildingBlock(net, "4f", 256, 1, 0.01, b4e);

  std::shared_ptr<Layer> b5a = BuildingBlock(net, "5a", 512, 2, 0.01, b4b);
  //std::shared_ptr<Layer> b5b = BuildingBlock(net, "5b", 512, 1, 0.01, b5a);
  BuildingBlock(net, "5b", 512, 1, 0.01, b5a);

  net.Add(GenPoolingConf("pool5", false, 7, 1, 0));
  net.Add(GenFlattenConf("flat"));
  net.Add(GenDenseConf("ip6-1", 1000, 0.01, 1));
  net.Add(GenDenseConf("ip6-2", 676, 0.01, 1));

  return net;
}

void TrainOneEpoch(FeedForwardNet &net, ILSVRC &data,
                   std::shared_ptr<Device> device, int epoch, string bin_folder,
                   size_t num_train_files, size_t batchsize, float lr,
                   Channel *train_ch, size_t pfreq, int nthreads) {
  float loss = 0.0f, metric = 0.0f;
  float load_time = 0.0f, train_time = 0.0f;
  size_t b = 0;
  size_t n_read;
  Timer timer, ttr;
  Tensor prefetch_x, prefetch_y;
  string binfile = bin_folder + "/train1.bin";
  timer.Tick();
  data.LoadData(kTrain, binfile, batchsize, &prefetch_x, &prefetch_y, &n_read,
                nthreads);
  load_time += timer.Elapsed();
  CHECK_EQ(n_read, batchsize);
  Tensor train_x(prefetch_x.shape(), device);
  Tensor train_y(prefetch_y.shape(), device, kInt);
  std::thread th;
  for (size_t fno = 1; fno <= num_train_files; fno++) {
    binfile = bin_folder + "/train" + std::to_string(fno) + ".bin";
    while (true) {
      if (th.joinable()) {
        th.join();
        load_time += timer.Elapsed();
        // LOG(INFO) << "num of samples: " << n_read;
        if (n_read < batchsize) {
          if (n_read > 0) {
            LOG(WARNING) << "Pls set batchsize to make num_total_samples "
                         << "% batchsize == 0. Otherwise, the last " << n_read
                         << " samples would not be used";
          }
          break;
        }
      }
      if (n_read == batchsize) {
        train_x.CopyData(prefetch_x);
        train_y.CopyData(prefetch_y);
      }
      timer.Tick();
      th = data.AsyncLoadData(kTrain, binfile, batchsize, &prefetch_x,
                              &prefetch_y, &n_read, nthreads);
      if (n_read < batchsize) continue;
      CHECK_EQ(train_x.shape(0), train_y.shape(0));
      ttr.Tick();
      auto ret = net.TrainOnBatch(epoch, train_x, train_y);
      train_time += ttr.Elapsed();
      loss += ret.first;
      metric += ret.second;
      b++;
    }
    if (b % pfreq == 0) {
      train_ch->Send(
          "Epoch " + std::to_string(epoch) + ", training loss = " +
          std::to_string(loss / b) + ", accuracy = " +
          std::to_string(metric / b) + ", lr = " + std::to_string(lr) +
          ", time of loading " + std::to_string(batchsize) + " images = " +
          std::to_string(load_time / b) +
          " ms, time of training (batchsize = " + std::to_string(batchsize) +
          ") = " + std::to_string(train_time / b) + " ms.");
      loss = 0.0f;
      metric = 0.0f;
      load_time = 0.0f;
      train_time = 0.0f;
      b = 0;
    }
  }
}

void TestOneEpoch(FeedForwardNet &net, ILSVRC &data,
                  std::shared_ptr<Device> device, int epoch, string bin_folder,
                  size_t num_test_images, size_t batchsize, Channel *val_ch,
                  int nthreads) {
  float loss = 0.0f, metric = 0.0f;
  float load_time = 0.0f, eval_time = 0.0f;
  size_t n_read;
  string binfile = bin_folder + "/test.bin";
  Timer timer, tte;
  Tensor prefetch_x, prefetch_y;
  timer.Tick();
  data.LoadData(kEval, binfile, batchsize, &prefetch_x, &prefetch_y, &n_read,
                nthreads);
  load_time += timer.Elapsed();
  Tensor test_x(prefetch_x.shape(), device);
  Tensor test_y(prefetch_y.shape(), device, kInt);
  int remain = (int)num_test_images - n_read;
  CHECK_EQ(n_read, batchsize);
  std::thread th;
  while (true) {
    if (th.joinable()) {
      th.join();
      load_time += timer.Elapsed();
      remain -= n_read;
      if (remain < 0) break;
      if (n_read < batchsize) break;
    }
    test_x.CopyData(prefetch_x);
    test_y.CopyData(prefetch_y);
    timer.Tick();
    th = data.AsyncLoadData(kEval, binfile, batchsize, &prefetch_x, &prefetch_y,
                            &n_read, nthreads);

    CHECK_EQ(test_x.shape(0), test_y.shape(0));
    tte.Tick();
    auto ret = net.EvaluateOnBatch(test_x, test_y);
    eval_time += tte.Elapsed();
    ret.first.ToHost();
    ret.second.ToHost();
    loss += Sum(ret.first);
    metric += Sum(ret.second);
  }
  loss /= num_test_images;
  metric /= num_test_images;
  val_ch->Send("Epoch " + std::to_string(epoch) + ", val loss = " +
               std::to_string(loss) + ", accuracy = " + std::to_string(metric) +
               ", time of loading " + std::to_string(num_test_images) +
               " images = " + std::to_string(load_time) +
               " ms, time of evaluating " + std::to_string(num_test_images) +
               " images = " + std::to_string(eval_time) + " ms.");
}

void Checkpoint(FeedForwardNet &net, string prefix) {
  Snapshot snapshot(prefix, Snapshot::kWrite, 200);
  auto names = net.GetParamNames();
  auto values = net.GetParamValues();
  for (size_t k = 0; k < names.size(); k++) {
    values.at(k).ToHost();
    snapshot.Write(names.at(k), values.at(k));
  }
  LOG(INFO) << "Write snapshot into " << prefix;
}

/*string GetLayerName(string param_name) {
  size_t pos = param_name.find("_weight");
  size_t length = param_name.length();
  if (pos > length)
    pos = param_name.find("_bias");
  if (pos > length)
    pos = param_name.find("_mean");
  if (pos > length)
    pos = param_name.find("_variance");
  if (pos > length)
    LOG(FATAL) << "Wrong param name: " << param_name;
  return param_name.substr(0, pos);
}

std::shared<ptr> FindLayerByName(FeedForwardNet &net, string name) {
  for (size_t i = 0; i < net.layers(); i++)
    if (net.layers().at(i).name() == name)
      return net.layers().at(i);
    return nullptr;
}*/

size_t GetIndexByName(vector<ParamSpec> specs, string name) {
  for (size_t i = 0; i < specs.size(); i++)
    if (specs[i].name() == name) return i;
  return specs.size();
}

void UpdateParam(Tensor& param, Tensor value) {
  CHECK_EQ(param.Size(), value.Size());
  param.CopyData(value);
}

void SetParams(ParamSpec& spec) {
  spec.set_lr_mult(0);
  spec.set_decay_mult(0);
}

void Train(int num_epoch, float lr, size_t batchsize, size_t train_file_size,
           string bin_folder, size_t num_train_images, size_t num_test_images,
           size_t pfreq, int nthreads, int state, string model) {
  ILSVRC data;
  data.ReadMean(bin_folder + "/mean.bin");
  FeedForwardNet net;
  if (state != 2) net = CreateNet();
  else net = CreateNet2();
  size_t nepoch = 0;

  auto cuda = std::make_shared<CudaGPU>(1);
  net.ToDevice(cuda);
  SGD sgd;
  OptimizerConf opt_conf;
  opt_conf.set_momentum(0.9);
  auto reg = opt_conf.mutable_regularizer();
  reg->set_coefficient(0.0001);
  sgd.Setup(opt_conf);
  if (state == 2)
    sgd.SetLearningRateGenerator(
      [lr](int epoch) { return lr * std::pow(0.1, epoch / 5); });
  else
    sgd.SetLearningRateGenerator(
      [lr](int epoch) { return lr * std::pow(0.1, epoch / 15); });

  SoftmaxCrossEntropy loss;
  Accuracy acc;
  net.Compile(true, &sgd, &loss, &acc);
  if (state) { // resume or finetune
    Snapshot snapshot(model, Snapshot::kRead, 200);
    size_t index = model.find("snapshot_epoch");
    nepoch = stoi(model.substr(index + 14));
    LOG(INFO) << "resume training from epoch: " << nepoch;
    auto ret = snapshot.Read();
    // auto names = net.GetParamNames();
    auto specs = net.GetParamSpecs();
    auto values = net.GetParamValues();
    // CHECK_EQ(names.size(), values.size());
    CHECK_EQ(specs.size(), values.size());
    for (size_t i = 0; i < ret.size(); i++) {
      string param_name = ret.at(i).first;
      Tensor nvalue = ret.at(i).second;
      LOG(INFO) << "parameter name: " << param_name;
      size_t idx = GetIndexByName(specs, param_name);
      LOG(INFO) << "index: " << idx;
      if (idx >= ret.size()) continue;
      // if not find corresponding layer, do not need to reload param
      LOG(INFO) << specs[idx].name() << " : " << values[idx].L1();
      UpdateParam(values[idx], nvalue);
      LOG(INFO) << specs[idx].name() << " : " << values[idx].L1();
      if (state == 2) {
        // finetune only last building block and fc layer
        // set lr of other layers to 0 to forbid learning
        vector<string> ft_layers = {"conv5a", "bn5a", "conv5b", "bn5b", "ip6"};
        for (size_t i = 0; i < ft_layers.size(); i++) {
          size_t tmp_idx = param_name.find(ft_layers[i]);
          if (tmp_idx < ft_layers[i].size()) break;
          if (i == ft_layers.size() - 1) SetParams(specs[idx]);
        }
      }
    }
  }

  Channel *train_ch = GetChannel("train_perf");
  train_ch->EnableDestStderr(true);
  Channel *val_ch = GetChannel("val_perf");
  val_ch->EnableDestStderr(true);
  size_t num_train_files = num_train_images / train_file_size +
                           (num_train_images % train_file_size ? 1 : 0);
  for (int epoch = (state == 1) ? nepoch+1 : 0; epoch < num_epoch; epoch++) {
    float epoch_lr = sgd.GetLearningRate(epoch);
    auto names = net.GetParamNames();
    auto values = net.GetParamValues();
    for (size_t k = 0; k < names.size(); k++)
      LOG(INFO) << names[k] << " : " << values[k].L1();
    TrainOneEpoch(net, data, cuda, epoch, bin_folder, num_train_files,
                  batchsize, epoch_lr, train_ch, pfreq, nthreads);
    if (epoch % ((state == 2)?5:10) == 0 && epoch > 0) {
      string prefix = "snapshot_epoch" + std::to_string(epoch);
      Checkpoint(net, prefix);
    }
    TestOneEpoch(net, data, cuda, epoch, bin_folder, num_test_images, batchsize,
                 val_ch, nthreads);
  }
}
}

int main(int argc, char **argv) {
  singa::InitChannel(nullptr);
  int pos = singa::ArgPos(argc, argv, "-h");
  if (pos != -1) {
    std::cout << "Usage:\n"
              << "\t-epoch <int>: number of epoch to be trained, default is 90;\n"
              << "\t-lr <float>: base learning rate;\n"
              << "\t-batchsize <int>: batchsize, it should be changed regarding "
                 "to your memory;\n"
              << "\t-filesize <int>: number of training images that stores in "
                 "each binary file;\n"
              << "\t-ntrain <int>: number of training images;\n"
              << "\t-ntest <int>: number of test images;\n"
              << "\t-data <folder>: the folder which stores the binary files;\n"
              << "\t-pfreq <int>: the frequency(in batch) of printing current "
                 "model status(loss and accuracy);\n"
              << "\t-nthreads <int>: the number of threads to load data which "
                 "feed to the model.\n"
              << "\t-state <string>: train or resume network\n"
              << "\t-model <folder>: the folder where net parameters"
                 " are stored\n";
    return 0;
  }
  pos = singa::ArgPos(argc, argv, "-epoch");
  int nEpoch = 80;
  if (pos != -1) nEpoch = atoi(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-lr");
  float lr = 0.05;
  if (pos != -1) lr = atof(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-batchsize");
  int batchsize = 256;
  if (pos != -1) batchsize = atof(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-filesize");
  size_t train_file_size = 1280;
  if (pos != -1) train_file_size = atoi(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-ntrain");
  size_t num_train_images = 921415; //1281167;
  if (pos != -1) num_train_images = atoi(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-ntest");
  size_t num_test_images = 50000;
  if (pos != -1) num_test_images = atoi(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-data");
  string bin_folder = "/home/xiangrui/jixin/alisc_data";
  if (pos != -1) bin_folder = argv[pos + 1];

  pos = singa::ArgPos(argc, argv, "-pfreq");
  size_t pfreq = 500;
  if (pos != -1) pfreq = atoi(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-nthreads");
  int nthreads = 12;
  if (pos != -1) nthreads = atoi(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-state");
  string state = "train";
  if (pos != -1) state = argv[pos + 1];

  pos = singa::ArgPos(argc, argv, "-snapshot");
  string snapshot = "snapshot_epoch50";
  if(pos != -1) snapshot = argv[pos + 1];

  LOG(INFO) << "Start training";
  singa::Train(nEpoch, lr, batchsize, train_file_size, bin_folder,
          num_train_images, num_test_images, pfreq, nthreads,
          (state == "resume") ? 1 : ((state == "finetune") ? 2 : 0), snapshot);
  LOG(INFO) << "End training";
}
#endif
