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

#ifndef SINGA_IO_READER_H_
#define SINGA_IO_READER_H_

#include <cstring>
#include <fstream>
#include <string>

namespace singa {
namespace io {

using std::string;

/// General Reader that provides functions for reading tuples.
/// Subclasses implement the functions for a specific data storage, e.g., CSV
/// file, HDFS, kvfile, leveldb, lmdb, etc.
class Reader {
 public:
  /// In case that users forget to call Close() to release resources, e.g.,
  /// memory, you can release them here.
  virtual ~Reader() {}

  /// path is the path to the storage, could be a file path, database
  /// connection, or hdfs path.
  /// return true if open successfully, otherwise false.
  virtual bool Open(const std::string& path, int capacity = 10485760) = 0;

  /// Release resources.
  virtual void Close() = 0;

  /// Read a tuple.
  /// return true if read successfully, otherwise false.
  virtual bool Read(std::string* key, std::string* value) = 0;

  /// Iterate through all tuples to get the num of all tuples.
  /// return num of tuples
  virtual int Count() = 0;
};

class BinFileReader : public Reader {
 public:
  ~BinFileReader() { Close(); }
  /// \copydoc Open(const std::string& path)
  bool Open(const std::string& path, int capacity = 10485760) override;
  /// \copydoc Close()
  void Close() override;
  /// \copydoc Read(std::string* key, std::string* value)
  bool Read(std::string* key, std::string* value) override;
  /// \copydoc Count()
  int Count() override;
  /// return path to binary file
  inline std::string path() { return path_; }

 protected:
  /// Read the next filed, including content_len and content;
  /// return true if succeed.
  bool ReadField(std::string* content);

  /// Read data from disk if the current data in the buffer is not a full field.
  /// size is the size of the next field.
  bool PrepareNextField(int size);

 private:
  std::string path_ = "";
  /// file to be read
  std::ifstream fdat_;
  /// internal buffer
  char* buf_ = nullptr;
  /// offset inside the buf_
  int offset_ = 0;
  /// allocated bytes for the buf_
  int capacity_ = 0;
  /// bytes in buf_
  int bufsize_ = 0;
  /// magic word
  const char kMagicWord[2] = {'s', 'g'};
};

}  // namespace io
}  // namespace singa

#endif  // SINGA_IO_READER_H_
