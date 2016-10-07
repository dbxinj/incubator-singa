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
#include <algorithm>
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

class FEATLOADER {
 public:
  FEATLOADER() {}
  ~FEATLOADER() {
    if (reader != nullptr) {
      reader->Close();
      delete reader;
    }
  }

  void LoadData(string file, size_t read_size, Tensor *x,
    size_t *n_read, int nthreads);

  std::thread AsyncLoadData(string file, size_t read_size, Tensor *x,
    size_t *n_read, int nthreads);

  void Transform(int thid, int nthreads, vector<string *> feats,
      Tensor *x);

  std::thread AsyncTransform(int thid, int nthreads,
              vector<string *> feats, Tensor *x);

 private:
  const size_t kFeatSize = 512;
  string last_read_file = "";
  BinFileReader *reader = nullptr;
};

std::thread FEATLOADER::AsyncLoadData(string file, size_t read_size, Tensor *x,
    size_t *n_read, int nthreads) {
  return std::thread(
      [=]() { LoadData(file, read_size, x, n_read, nthreads); });
}

void FEATLOADER::LoadData(string file, size_t read_size, Tensor *x,
    size_t *n_read, int nthreads) {
  x->Reshape(Shape{read_size, kFeatSize});
  if (file != last_read_file) {
    if (reader != nullptr) {
      reader->Close();
      delete reader;
      reader = nullptr;
    }
    reader = new BinFileReader();
    reader->Open(file, 100 << 20);
    last_read_file = file;
  } else if (reader == nullptr) {
    reader = new BinFileReader();
    reader->Open(file, 100 << 20);
  }
  vector<string *> feats;
  for (size_t i = 0; i < read_size; i++) {
    string *feat = new string();
    string path;
    bool ret = reader->Read(&path, feat);
    if(ret == false) {
      reader->Close();
      delete reader;
      reader = nullptr;
      break;
    }
    feats.push_back(feat);
  }
  int nfeat = feats.size();
  LOG(INFO) << "nfeat: " << nfeat;
  *n_read = nfeat;

  vector<std::thread> threads;
  for (int i = 1; i < nthreads; i++) {
    threads.push_back(AsyncTransform(i, nthreads, feats, x));
  }
  Transform(0, nthreads, feats, x);
  for (size_t i = 0; i < threads.size(); i++) threads[i].join();
  for (int k = 0; k < nfeat; k++) delete feats.at(k);
}

std::thread FEATLOADER::AsyncTransform(int thid, int nthreads,
              vector<string *> feats, Tensor *x) {
  return std::thread(
      [=]() { Transform(thid, nthreads, feats, x); });
}

void FEATLOADER::Transform(int thid, int nthreads, vector<string *> feats,
      Tensor *x) {
  int nfeat = feats.size();
  int start = nfeat / nthreads * thid;
  int end = start + nfeat / nthreads;
  //LOG(INFO) << "length: " << (*feats.at(start)).length();
  size_t dsize = (*feats.at(start)).length() / (sizeof(float) / sizeof(char));
  for (int k = start; k < end; k++) {
    const char* c = (*feats.at(k)).c_str();
    const float* d = reinterpret_cast<const float*>(c);
    x->CopyDataFromHostPtr<float>(d, dsize, k * dsize);
  }
  if (!thid) {
    for (int k = nfeat / nthreads * nthreads; k < nfeat; k++) {
      const char* c = (*feats.at(k)).c_str();
      const float* d = reinterpret_cast<const float*>(c);
      x->CopyDataFromHostPtr<float>(d, dsize, k * dsize);
    }
  }
}

void ComputeDist(int thid, int nthreads, Tensor a, Tensor b, Tensor* c) {
  // a{nq, dim}, b{ndb, dim}, c{nq, ndb}
  size_t nquery = a.shape(0); // number of queries
  size_t start = nquery / nthreads * thid;
  size_t end = start + nquery / nthreads;
  size_t dim = a.shape(1); // feature dim
  size_t ndb = b.shape(0); // number of db features
  c->Reshape(Shape{nquery, ndb});
  CHECK_EQ(a.shape(1), b.shape(1));
  //LOG(INFO) << "feat dim: " << dim << ", ndb: " << ndb;
  //LOG(INFO) << "start: " << start << ", end: " << end;
  //const float* data = a.data<float>();
  for (size_t k = start; k < end; k++) {
    Tensor q(Shape{dim});
    Tensor out(Shape{ndb});
    CopyDataToFrom(&q, a, dim, 0, k * dim);
    //LOG(INFO) << "get query data";
    Tensor diff = b;
    SubRow(q, &diff);
    SumColumns(Square(diff), &out);
    //LOG(INFO) << "get dist.";
    CopyDataToFrom(c, out, ndb, k * ndb, 0);
  }
  if (!thid) {
    for (size_t k = nquery / nthreads * nthreads; k < nquery; k++) {
      Tensor q(Shape{dim}), out(Shape{ndb});
      CopyDataToFrom(&q, a, dim, 0, k * dim);
      Tensor diff = b;
      SubRow(q, &diff);
      SumColumns(Square(diff), &out);
      CopyDataToFrom(c, out, ndb, k * ndb, 0);
    }
  }
}

void ComputeDist(Tensor& a, Tensor& b, Tensor* c) {
  size_t nquery = a.shape(0), dim = a.shape(1), ndb = b.shape(0);
  c->Reshape(Shape{nquery, ndb});
  const float* data = a.data<float>();
  for (size_t i = 0; i < nquery; i++) {
    Tensor q(Shape{dim}), out(Shape{ndb});
    q.CopyDataFromHostPtr(data + i * dim, dim);
    Tensor diff(b.shape());
    diff.CopyData(b);
    SubRow(q, &diff);
    SumColumns(Square(diff), &out);
    CopyDataToFrom(c, out, ndb, i * ndb, 0);
  }
}

std::thread AsyncComputeDist(int thid, int nthreads, Tensor a, Tensor b, Tensor* c) {
  return std::thread(
      [=]() { ComputeDist(thid, nthreads, a, b, c); });
}

void AppendDist(Tensor& a, Tensor* b, size_t offset) {
  size_t query = a.shape(0), bz = a.shape(1), ndb = b->shape(0);
  for (size_t i = 0; i < query; i++)
    CopyDataToFrom(b, a, bz, i * ndb + offset, i * bz);
}

void SortDist(Tensor& ret, vector<vector<size_t>> *idx, vector<vector<float>> *dist) {
  size_t nquery = ret.shape(0);
  size_t ndb = ret.shape(1);
  LOG(INFO) << "query: " << nquery << ", db: " << ndb;
  const float* data = ret.data<float>();
  LOG(INFO) << "res: " << data[0];
  for (size_t i = 0; i < nquery; i++) {
    vector<float> d(data + i * ndb, data + (i + 1) * ndb);
    vector<size_t> index(ndb);
    iota(index.begin(), index.end(), 0);

    // sort idx based on value d
    std::sort(std::begin(index), std::end(index),
        [&] (float d1, float d2) { return d[d1] < d[d2]; });

    LOG(INFO) << "i: " << i;
    for (size_t j = 0; j < ndb; j++) {
      if (index[j] < 20) {
        idx->at(i)[index[j]] = j;
        dist->at(i)[index[j]] = d[j];
        if(!i) LOG(INFO) << "index y: " << index[j]
          << "idx is: " << j << ", dist:" << d[j];
      }
    }
  }
}

void LoadImgList(string image_list, vector<string>* names) {
  std::ifstream image_list_file(image_list.c_str(), std::ios::in);
  string name;
  while (image_list_file >> name)
    names->push_back(name.substr(0, name.length() - 4)); // remove '.jpg'
}

void WriteResults(vector<vector<float>> &dis, vector<vector<size_t>> &idx,
    string tar_file, string query_list_file, string image_list_file) {
  std::ofstream file;
  file.open(tar_file);
  vector<string> img_names;
  vector<string> q_names;
  LoadImgList(image_list_file, &img_names);
  LoadImgList(query_list_file, &q_names);
  size_t k = dis[0].size();
  for (size_t i = 0; i < q_names.size(); i++) {
    string out = "";
    out += q_names[i] + ",";
    for (size_t j = 0; j < k; j++)
      out += img_names[idx[i][j]] + ":" + std::to_string(dis[i][j]) + ";";
    out += "\n";
    file << out;
  }
  file.close();
}

void Retrieve(size_t batchsize, size_t feat_file_size, string bin_folder,
              size_t num_eval_files, string query_file, string img_file,
              string tar_file, int nthreads) {
  FEATLOADER loader;
  float load_time = 0.f, search_time = 0.f;
  size_t n_read, batch_num = 0;
  size_t query = 100, total = 20;//1069124;
  string binfile = bin_folder + "/feat1.bin";
  string qbinfile = bin_folder + "/qfeat.bin";
  Timer timer, ttr;
  timer.Tick();
  Tensor prefetch_q, prefetch_x, ret(Shape{query, total});
  loader.LoadData(qbinfile, query, &prefetch_q, &n_read, nthreads);
  loader.LoadData(binfile, batchsize, &prefetch_x, &n_read, nthreads);
  Tensor eval_x(prefetch_x.shape());
  LOG(INFO) << "shape q: " << prefetch_q.shape(0) << prefetch_q.shape(1);
  LOG(INFO) << "shape x: " << prefetch_x.shape(0) << prefetch_x.shape(1);
  // const float* data = prefetch_x.data<float>();
  //LOG(INFO) << "first 3 data: " << data[0] << data[1] << data[2];
  load_time += timer.Elapsed();
  std::thread th;
  LOG(INFO) << "number of evaluation files: " << num_eval_files;
  for (size_t fno = 1; fno <= num_eval_files; fno++) {
    binfile = bin_folder + "/feat" + std::to_string(fno) + ".bin";
    while(true) {
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
      timer.Tick();
      if (n_read == batchsize) eval_x.CopyData(prefetch_x);
      //th = loader.AsyncLoadData(binfile, batchsize, &prefetch_x, &n_read, nthreads);
      if (n_read < batchsize) continue;
      ttr.Tick();

      // compute distnace
      vector<std::thread> threads;
      Tensor tmp{Shape(query, n_read)};
      /*for (int i = 1; i < nthreads; i++)
        threads.push_back(AsyncComputeDist(i, nthreads,
              prefetch_q, eval_x, &tmp));
      ComputeDist(0, nthreads, prefetch_q, eval_x, &tmp);
      for (size_t i = 0; i < threads.size(); i++) threads[i].join();*/
      ComputeDist(prefetch_q, eval_x, &tmp);
      const float* dist = tmp.data<float>();
      LOG(INFO) << "dist:" << dist[0];
      AppendDist(tmp, &ret, (batch_num++) * batchsize);
      search_time += ttr.Elapsed();
      break;
    }
  if (n_read < batchsize && n_read > 0) {
    ttr.Tick();
    LOG(INFO) << "last batch: " << prefetch_x.shape(1);
    Tensor x(Shape{n_read, prefetch_x.shape(1)});
    CopyDataToFrom(&x, prefetch_x, x.Size(), 0, 0);

    // compute distnace of the last batch
    vector<std::thread> threads;
    Tensor tmp(Shape{query, n_read});
    for (int i = 1; i < nthreads; i++)
      threads.push_back(AsyncComputeDist(i, nthreads,
            prefetch_q, x, &tmp));
    ComputeDist(0, nthreads, prefetch_q, x, &tmp);
    for (size_t i = 0; i < threads.size(); i++) threads[i].join();

    LOG(INFO) << "compute dist done.";
    AppendDist(tmp, &ret, batch_num * batchsize);
    search_time += ttr.Elapsed();
  }
  }
    if (n_read < batchsize && n_read > 0) {
    ttr.Tick();
    LOG(INFO) << "last batch: " << prefetch_x.shape(1);
    Tensor x(Shape{n_read, prefetch_x.shape(1)});
    CopyDataToFrom(&x, prefetch_x, x.Size(), 0, 0);

    // compute distnace of the last batch
    vector<std::thread> threads;
    Tensor tmp(Shape{query, n_read});
    for (int i = 1; i < nthreads; i++)
      threads.push_back(AsyncComputeDist(i, nthreads,
            prefetch_q, x, &tmp));
    ComputeDist(0, nthreads, prefetch_q, x, &tmp);
    for (size_t i = 0; i < threads.size(); i++) threads[i].join();

    LOG(INFO) << "compute dist done.";
    AppendDist(tmp, &ret, batch_num * batchsize);
    search_time += ttr.Elapsed();
  }
  //float dis[query][20] = {0.f};
  //size_t idx[query][20] = {0};
  vector<vector<float>> dis(query, vector<float>(20, 0.f));
  vector<vector<size_t>> idx(query, vector<size_t>(20, 0));
  SortDist(ret, &idx, &dis);
  WriteResults(dis, idx, tar_file, query_file, img_file);
  LOG(INFO) << "write result to file " << tar_file;

  for (size_t i = 0; i < query; i++) {
    dis[i].erase(dis[i].begin(), dis[i].begin() + 20);
    idx[i].erase(idx[i].begin(), idx[i].begin() + 20);
  }
  dis.erase(dis.begin(), dis.begin() + query);
  idx.erase(idx.begin(), idx.begin() + query);
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
  int batchsize = 10000;
  if (pos != -1) batchsize = atof(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-filesize");
  size_t feat_file_size = 1280;
  if (pos != -1) feat_file_size = atoi(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-neval");
  size_t num_eval_files = 1;//1069124; //1281167;
  if (pos != -1) num_eval_files = atoi(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-data");
  string bin_folder = "/home/xiangrui/jixin/alisc_feat";
  if (pos != -1) bin_folder = argv[pos + 1];

  pos = singa::ArgPos(argc, argv, "-qfile");
  string query_file = "/home/xiangrui/jixin/alisc15/tri_val_q.txt";
  if(pos != -1) query_file = argv[pos + 1];

  pos = singa::ArgPos(argc, argv, "-imgfile");
  string img_file = "/home/xiangrui/jixin/alisc15/eval_list.txt";
  if(pos != -1) img_file = argv[pos + 1];

  pos = singa::ArgPos(argc, argv, "-tarfile");
  string tar_file = "/home/xiangrui/jixin/alisc15/result.txt";
  if(pos != -1) tar_file = argv[pos + 1];

  pos = singa::ArgPos(argc, argv, "-pfreq");
  size_t pfreq = 100;
  if (pos != -1) pfreq = atoi(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-nthreads");
  int nthreads = 12;
  if (pos != -1) nthreads = atoi(argv[pos + 1]);

  LOG(INFO) << "Start retrieval";
  singa::Retrieve(batchsize, feat_file_size, bin_folder,
          num_eval_files, query_file, img_file, tar_file, nthreads);
  LOG(INFO) << "End retrieval";
}
#endif
