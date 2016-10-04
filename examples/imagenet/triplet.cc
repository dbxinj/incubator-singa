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
#include "triplet.h"
#include "singa/utils/channel.h"
#include "singa/utils/string.h"
int main(int argc, char **argv) {
  int pos = singa::ArgPos(argc, argv, "-h");
  if (pos != -1) {
    std::cout << "Usage:\n"
              << "\t-trainlist <file>: the file of training list;\n"
              << "\t-trainfolder <folder>: the folder of training images;\n"
              << "\t-testlist <file>: the file of test list;\n"
              << "\t-testfolder <floder>: the folder of test images;\n"
              << "\t-outdata <folder>: the folder to save output files;\n"
              << "\t-filesize <int>: number of training images that stores in "
                 "each binary file.\n";
    return 0;
  }
  pos = singa::ArgPos(argc, argv, "-trainlist");
  string train_image_list = "/home/xiangrui/jixin/alisc15/tri_train.txt";
  if (pos != -1) train_image_list = argv[pos + 1];

  pos = singa::ArgPos(argc, argv, "-trainfolder");
  string train_image_folder = "/home/xiangrui/jixin/alisc15/eval_image";
  if (pos != -1) train_image_folder = argv[pos + 1];

  pos = singa::ArgPos(argc, argv, "-testlist");
  string test_image_list = "/home/xiangrui/jixin/alisc15/tri_val.txt";
  if (pos != -1) test_image_list = argv[pos + 1];

  pos = singa::ArgPos(argc, argv, "-testfolder");
  string test_image_folder = "/home/xiangrui/jixin/alisc15/query_image";
  if (pos != -1) test_image_folder = argv[pos + 1];

  pos = singa::ArgPos(argc, argv, "-outdata");
  string bin_folder = "/home/xiangrui/jixin/alisc_triplet_data";
  if (pos != -1) bin_folder = argv[pos + 1];

  pos = singa::ArgPos(argc, argv, "-filesize");
  size_t train_file_size = 1280;
  if (pos != -1) train_file_size = atoi(argv[pos + 1]);

  singa::TRIPLET data;
  LOG(INFO) << "Creating training and test data...";
  //data.CreateTrainData(train_image_list, train_image_folder, bin_folder,
  //                     train_file_size);
  //data.CreateEvalData(train_image_list, train_image_folder, bin_folder,
  //                     train_file_size);
  data.CreateTestData(test_image_list, test_image_folder, bin_folder);
  LOG(INFO) << "Data created!";
  return 0;
}
#endif  // USE_OPENCV
