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
#include <string>
#include "./triplet.h"
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
  net.Add(GenDenseConf("ip6", 676, 0.01, 1));

  return net;
}

string Serialize(Tensor data) {
  LOG(INFO) << "data size:" << data.Size();
  LOG(INFO) << "data shape: " << data.shape(0) << data.shape(1);
  string *out = new string;
  const float *d = data.data<float>();
  LOG(INFO) << "data size:" << data.Size();
  const char *s = reinterpret_cast<const char*>(d);
  memcpy(out, s, data.Size() * sizeof(float) / sizeof(char));
  return *out;
}

void ExtractOnBatch(FeedForwardNet &net, TRIPLET &data,
                   std::shared_ptr<Device> device, string bin_folder,
                   size_t num_eval_files, size_t batchsize, string layer_name,
                   string tar_folder, size_t pfreq, int nthreads) {
  float load_time = 0.0f, extract_time = 0.0f;
  size_t b = 0;
  size_t n_read;
  size_t num_extract_images = 0;
  BinFileWriter *bfwriter = new BinFileWriter();
  size_t file_num = 1u;
  string outfile = tar_folder + "/feat" + std::to_string(file_num++) + ".bin";
  bfwriter->Open(outfile, kCreate);
  Timer timer, ttr;
  Tensor prefetch_x, prefetch_y;
  string binfile = bin_folder + "/eval1.bin";
  timer.Tick();
  data.LoadData(kEval, binfile, batchsize, &prefetch_x, &prefetch_y, &n_read,
                nthreads);
  load_time += timer.Elapsed();
  CHECK_EQ(n_read, batchsize);
  Tensor eval_x(prefetch_x.shape(), device);
  //Tensor train_y(prefetch_y.shape(), device, kInt);
  std::thread th;
  LOG(INFO) << "number of eval files: " << num_eval_files;
  for (size_t fno = 1; fno <= num_eval_files; fno++) {
    binfile = bin_folder + "/eval" + std::to_string(fno) + ".bin";
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
        eval_x.CopyData(prefetch_x);
        //train_y.CopyData(prefetch_y);
      }
      timer.Tick();
      th = data.AsyncLoadData(kEval, binfile, batchsize, &prefetch_x,
                              &prefetch_y, &n_read, nthreads);
      if (n_read < batchsize) continue;
      //CHECK_EQ(train_x.shape(0), train_y.shape(0));
      ttr.Tick();
      //auto ret = net.TrainOnBatch(epoch, train_x, train_y);
      //LOG(INFO) << "layer name : " << layer_name;
      Tensor fea = net.Extract(eval_x, layer_name);
      fea.ToHost();
      //LOG(INFO) << "feature shape dim: " << fea.nDim();
      extract_time += ttr.Elapsed();
      // LOG(INFO) << "extract " << batchsize << " image features";
      num_extract_images += batchsize;
      //char* out = new char[feat.Size() * 4];
      //string* out = new string;
      const float *data = fea.data<float>();
      //LOG(INFO) << "data size:" << fea.Size();
      //LOG(INFO) << "data shape:" << fea.shape(0) << fea.shape(1);
      //const char *s = reinterpret_cast<const char*>(ddata);
      //memcpy(&out, s, feat.Size() * sizeof(float) / sizeof(char));
      Shape shape = fea.shape();
      shape.at(0) /= fea.shape(0);
      size_t size = fea.Size() / fea.shape(0);
      for (size_t i = 0; i < fea.shape(0); i++) {
        Tensor tmp(shape);
        tmp.CopyDataFromHostPtr<float>(data + i * size, size);
        const float *sdata = tmp.data<float>();
        //if(!i) LOG(INFO) << "first 3 data: " << sdata[0] << sdata[1] << sdata[2];
        const char *s = reinterpret_cast<const char*>(sdata);
        string out(s, size * sizeof(float) / sizeof(char));
        //memcpy(out, s, fea.Size() * sizeof(float) / sizeof(char));
        if (bfwriter == nullptr) {
          bfwriter = new BinFileWriter();
          outfile = tar_folder + "/feat" + std::to_string(file_num++) + ".bin";
          bfwriter->Open(outfile, kCreate);
        }
        bfwriter->Write(tar_folder, out);
        //delete[] out;
      }
      b++;
    }
    if (b % pfreq == 0) {
      LOG(INFO) << "time of loading " << std::to_string(batchsize)
                << " images = " << std::to_string(load_time / b)
                << " ms, time of feature extraction (batchsize = "
                << std::to_string(batchsize) << ") = "
                << std::to_string(extract_time / b) << " ms.";
      load_time = 0.0f;
      extract_time = 0.0f;
      b = 0;
      bfwriter->Flush();
      bfwriter->Close();
      delete bfwriter;
      bfwriter = nullptr;
    }
  }
  if (n_read < batchsize && n_read > 0) {
    eval_x.SetValue(0.f);
    const float* d = prefetch_x.data<float>();
    size_t size = prefetch_x.Size();
    eval_x.CopyDataFromHostPtr(d, size);
    Tensor fea = net.Extract(eval_x, layer_name);
    fea.ToHost();
    LOG(INFO) << "extract " << n_read << " features";
    num_extract_images += n_read;
    //string *out = new string;
    const float *data = fea.data<float>();
    LOG(INFO) << "data size:" << fea.Size();
    Shape shape = fea.shape();
    shape.at(0) /= fea.shape(0);
    size = fea.Size() / fea.shape(0);
    for (size_t i = 0; i < prefetch_x.shape(0); i++) {
      Tensor tmp(shape);
      tmp.CopyDataFromHostPtr<float>(data + i * size, size);
      const float* sdata = tmp.data<float>();
      const char *s = reinterpret_cast<const char*>(sdata);
      string out(s, size * sizeof(float) / sizeof(char));
      //memcpy(out, s, fea.Size() * sizeof(float) / sizeof(char));

      bfwriter->Write(tar_folder, out);
      //delete[] out;
    }
  }
  if (bfwriter != nullptr) {
    bfwriter->Flush();
    bfwriter->Close();
    delete bfwriter;
    bfwriter = nullptr;
  }
  LOG(INFO) << "extract " << num_extract_images << " features done.";
  LOG(INFO) << "save features to " << tar_folder;
}

void ExtractQuery(FeedForwardNet &net, TRIPLET &data,
                   std::shared_ptr<Device> device, string bin_folder,
                   string layer_name, string tar_folder,
                   int nthreads) {
  float load_time = 0.0f;
  size_t n_read;
  size_t num_extract_images = 0;
  BinFileWriter *bfwriter = new BinFileWriter();
  string outfile = tar_folder + "/qfeat.bin";
  LOG(INFO) << "output file: " << outfile;
  bfwriter->Open(outfile, kCreate);
  Timer timer, ttr;
  Tensor prefetch_x, prefetch_y;
  string binfile = bin_folder + "/query.bin";
  timer.Tick();
  data.LoadData(kEval, binfile, 100, &prefetch_x, &prefetch_y, &n_read,
                nthreads);
  CHECK_EQ(100, n_read);
  load_time += timer.Elapsed();
  Tensor eval_x(prefetch_x.shape(), device);
  eval_x.CopyData(prefetch_x);
  LOG(INFO) << "get query data.";
  Tensor fea = net.Extract(eval_x, layer_name);
  fea.ToHost();
  const float* d = fea.data<float>();
  num_extract_images += n_read;
  Shape shape = fea.shape();
  shape.at(0) /= fea.shape(0);
  size_t size = fea.Size() / fea.shape(0);
  LOG(INFO) << "size: " << size;
  for (size_t i = 0; i < fea.shape(0); i++) {
    Tensor tmp(shape);
    tmp.CopyDataFromHostPtr<float>(d + i * size, size);
    const float* sdata = tmp.data<float>();
    const char *s = reinterpret_cast<const char*>(sdata);
    string out(s, size * sizeof(float) / sizeof(char));
    //memcpy(out, s, fea.Size() * sizeof(float) / sizeof(char));

    bfwriter->Write(tar_folder, out);
    //delete[] out;
  }
  bfwriter->Flush();
  bfwriter->Close();
  delete bfwriter;
  LOG(INFO) << "extract " << num_extract_images << " features done.";
  LOG(INFO) << "save features to " << tar_folder;
}

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

void Extract(size_t batchsize, size_t train_file_size, string bin_folder,
           size_t num_eval_images, string& layer_name, string tar_folder,
           size_t pfreq, int nthreads, string snapshot) {
  TRIPLET data;
  data.ReadMean(bin_folder + "/mean.bin");
  FeedForwardNet net = CreateNet();
  // size_t nepoch = 0;

  auto cuda = std::make_shared<CudaGPU>(1);
  net.ToDevice(cuda);
  SGD sgd;
  OptimizerConf opt_conf;
  opt_conf.set_momentum(0.9);
  auto reg = opt_conf.mutable_regularizer();
  reg->set_coefficient(0.0001);
  sgd.Setup(opt_conf);

  SoftmaxCrossEntropy loss;
  Accuracy acc;
  net.Compile(true, &sgd, &loss, &acc);

  Snapshot snap_shot(snapshot, Snapshot::kRead, 200);
  // size_t index = model.find("snapshot_epoch");
  // nepoch = stoi(model.substr(index + 14));
  // LOG(INFO) << "resume training from epoch: " << nepoch;
  auto ret = snap_shot.Read();
  // auto names = net.GetParamNames();
  auto specs = net.GetParamSpecs();
  auto values = net.GetParamValues();
  // CHECK_EQ(names.size(), values.size());
  CHECK_EQ(specs.size(), values.size());
  for (size_t i = 0; i < ret.size(); i++) {
    string param_name = ret.at(i).first;
    Tensor nvalue = ret.at(i).second;
    //LOG(INFO) << "parameter name: " << param_name;
    size_t idx = GetIndexByName(specs, param_name);
    //LOG(INFO) << "index: " << idx;
    if (idx >= specs.size()) continue;
    // if not find corresponding layer, do not need to reload param
    LOG(INFO) << specs[idx].name() << " : " << values[idx].L1();
    UpdateParam(values[idx], nvalue);
    LOG(INFO) << specs[idx].name() << " : " << values[idx].L1();
  }

  size_t num_eval_files = num_eval_images / train_file_size +
                           (num_eval_images % train_file_size ? 1 : 0);

  LOG(INFO) << "target file: " << tar_folder;
  LOG(INFO) << "snapshot : " << snapshot;
  LOG(INFO) << "layer name : " << layer_name;
  LOG(INFO) << "number of evaluate files: " << num_eval_files;
  ExtractOnBatch(net, data, cuda, bin_folder, num_eval_files,
                  batchsize, layer_name, tar_folder, pfreq, nthreads);
  ExtractQuery(net, data, cuda, bin_folder, layer_name, tar_folder, nthreads);
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
  pos = singa::ArgPos(argc, argv, "-batchsize");
  int batchsize = 256;
  if (pos != -1) batchsize = atof(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-filesize");
  size_t train_file_size = 1280;
  if (pos != -1) train_file_size = atoi(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-neval");
  size_t num_eval_images = 1069124; //1281167;
  if (pos != -1) num_eval_images = atoi(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-data");
  string bin_folder = "/home/xiangrui/jixin/alisc_eval_data";
  if (pos != -1) bin_folder = argv[pos + 1];

  pos = singa::ArgPos(argc, argv, "-layer");
  string layer_name = "flat";
  if (pos != -1) layer_name = argv[pos + 1];

  pos = singa::ArgPos(argc, argv, "-tarfile");
  string tar_folder = "/home/xiangrui/jixin/alisc_feat";
  if(pos != -1) tar_folder = argv[pos + 1];

  pos = singa::ArgPos(argc, argv, "-pfreq");
  size_t pfreq = 100;
  if (pos != -1) pfreq = atoi(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-nthreads");
  int nthreads = 12;
  if (pos != -1) nthreads = atoi(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-snapshot");
  string snapshot = "snapshot_epoch35";
  if(pos != -1) snapshot = argv[pos + 1];

  LOG(INFO) << "Start training";
  singa::Extract(batchsize, train_file_size, bin_folder,
          num_eval_images, layer_name, tar_folder, pfreq, nthreads, snapshot);
  LOG(INFO) << "End training";
}
#endif
