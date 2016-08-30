# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
""" CIFAR10 dataset is at https://www.cs.toronto.edu/~kriz/cifar.html.
It includes 5 binary dataset, each contains 10000 images. 1 row (1 image)
includes 1 label & 3072 pixels.  3072 pixels are 3 channels of a 32x32 image
"""

import cPickle
import numpy as np
import os
import argparse
import random
from struct import unpack
import cv2
from multiprocessing import Pool
from itertools import repeat, izip
import time

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))
from singa import utils
from singa import optimizer
from singa import device
from singa import tensor
from singa.proto import core_pb2, io_pb2

import resnet

def dim_order_transform(src_type, data):
    '''
    Args:
        src_type: transform data dim order from src_type to its opposite type,
                  i.e., 'CHW' to 'HWC', 'HWC' or 'CHW'
    Note:
        dim order has only two valid value, namely 'CHW' or 'CWH'
    '''
    data = np.array(data)
    shape = data.shape
    if src_type.upper() == 'CHW':
        '''new_shape = (shape[1], shape[2], shape[0])
        new = np.zeros(new_shape, dtype=np.float32)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    new[j][k][i] = data[i][j][k] * 1.0'''
        new = np.swapaxes(np.swapaxes(data, 0, 2), 0, 1)
    elif src_type.upper() == 'HWC':
        '''new_shape = (shape[2], shape[0], shape[1])
        new = np.zeros(new_shape, dtype=np.float32)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    new[k][i][j] = data[i][j][k] * 1.0'''
        new = np.swapaxes(np.swapaxes(data, 0, 2), 1, 2)
    else:
        assert False, "Invalid dim order."
    return new

def multi_run_wrapper(args):
    return data_transform(*args)

def data_transform(flag, data, mean, crop_size):
    input, label = parsing(data)
    input = dim_order_transform('HWC', input)
    input -= mean
    shape = input.shape
    if flag == 1: 
        # train phase, random crop and mirror
        h_offset = random.randint(0, (shape[1]-crop_size)/2)
        w_offset = random.randint(0, (shape[2]-crop_size)/2)
        output = crop(input, crop_size, crop_size, h_offset, w_offset)
        if random.random() < 0.5:
            output = mirror(output, True, False)
    else:
        # eval phase, central crop and no mirror
        output = crop(input, crop_size, crop_size, (shape[1]-crop_size)/2, 
            (shape[2]-crop_size)/2)
        
    return output, label

def crop(input, crop_h, crop_w, h_offset, w_offset):
    # default value: dim order is 'CHW', channel is 3
    shape = input.shape
    assert len(shape) != 4, "Not implemented yet."
    output = np.zeros((shape[0], crop_h, crop_w), dtype=np.float32)
    #output = tensor.Tensor((shape[0], crop_h, crop_w), input.device)
    for i in range(shape[0]):
        for j in range(crop_h):
            output[i][j] = input[i][h_offset+j][w_offset:(w_offset+crop_w)]
    return output

def mirror(input, h_mirror, v_mirror):
    # default dim order is 'CHW'
    if h_mirror == False and v_mirror == False:
        return input
    shape = input.shape
    assert len(shape) != 4, "Not implemented yet."
    output = np.zeros(shape, dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if h_mirror == True and v_mirror == True:
                    output[i][j][k] = input[i][shape[1]-j-1][shape[2]-k-1]
                elif h_mirror == True:
                    output[i][j][k] = input[i][j][shape[2]-k-1]
                else:
                    output[i][j][k] = input[i][shape[1]-j-1][k]
    return output

def load_mean(filepath):
    print 'Loading mean image from data file %s/mean.bin' % filepath
    f = open(filepath + '/mean.bin', 'rb')
    f.read(2)
    d = ord(f.read(1))
    if d == 0:
        f.read(1)
        vsize = unpack('I', f.read(4))[0]
    else:
        f.read(1)
        ksize = unpack('I', f.read(4))[0]
        print 'key size:', ksize
        print 'key:', f.read(ksize+4)
        vsize = unpack('I', f.read(4))[0]
        print 'value size:', vsize
    image_mean = io_pb2.ImageRecord()
    image_mean.ParseFromString(str(f.read(vsize+4)))
    pixels = image_mean.pixel
    mean = np.fromstring(pixels, np.uint8)
    mat = cv2.imdecode(mean, cv2.CV_LOAD_IMAGE_COLOR)
    if type(mat) == type(None):
        return np.array([])
    print 'shape of mean image: ', np.array(mat).shape
    #print np.array(mat).shape
    mat = dim_order_transform('HWC', mat)
    print 'shape of transformed mean image: ', mat.shape
    return mat

def load_batch(f, batch_size):
    '''
        each sample is stored as the following format: sg10 (4Bytes), [key size, key], value size, value
    '''
    for i in range(batch_size):
        if i == 0:
            images = []
        f.read(2)
        d = f.read(1)
        if d == '':
            return images
        if ord(d) == 0:
            f.read(1)
            vsize = unpack('I', f.read(4))[0]
        else:
            f.read(1)
            ksize = unpack('I', f.read(4))[0]
            f.read(ksize+4)
            vsize = unpack('I', f.read(4))[0]
        images.append(f.read(vsize+4))
    return images

def parsing(data):
    image = io_pb2.ImageRecord()
    image.ParseFromString(data)
    pixels = image.pixel
    pixels = np.fromstring(pixels, np.uint8)
    mat = cv2.imdecode(pixels, cv2.CV_LOAD_IMAGE_COLOR)
    if type(mat) == type(None):
        return np.array([]), np.array([])
    return mat, image.label[0]

def resnet_lr(epoch):
    if epoch < 80:
        return 0.1
    elif epoch < 120:
        return 0.01
    else:
        return 0.001

def train_one_epoch(net, opt, get_lr, epoch, bin_folder, num_train_batch,
         num_train_files, batch_size, mean, dev):
    loss, acc = 0.0, 0.0
    binfile = bin_folder + '/train1.bin'
    fp = open(binfile, 'rb')
    fnum = 1
    for b in range(num_train_batch):
        print 'num training batch:', b
        tx = tensor.Tensor((batch_size, 3, 224, 224), dev)
        ty = tensor.Tensor((batch_size,), dev, core_pb2.kInt)
        x = []
        y = []
        start = time.time()
        data = load_batch(fp, batch_size)
        pool = Pool(processes=24)
        tmp = pool.map(multi_run_wrapper, izip(repeat(1), data, repeat(mean), repeat(224)))
        pool.close()
        pool.join() 
        if len(tmp) > 0:
            tmpx, tmpy = zip(*tmp)
        if len(tmp) < batch_size:
            if len(tmp) > 0:
                x.extend(tmpx)
                y.extend(tmpy)
            fp.close()
            #print 'Loading done for binfile: train', fnum, '.bin'
            fnum += 1
            if fnum > num_train_files:
                return loss, acc
            binfile = bin_folder + '/train' + str(fnum) + '.bin'
            fp = open(binfile, 'rb')
            data = load_batch(fp, batch_size-len(x))
            pool = Pool(processes=24)
            tmp = pool.map(multi_run_wrapper, izip(repeat(1), data, repeat(mean), repeat(224)))
            pool.close()
            pool.join()
            tmpx, tmpy = zip(*tmp)
        '''for i in range(batch_size):
            tmpx, tmpy = data_transform(1, data[i], mean, 224)
            x.append(tmpx)
            y.append(tmpy)'''

        x.extend(tmpx)
        y.extend(tmpy)
        print 'pre-processing time:', time.time() - start
        start = time.time()
        tx.copy_from_numpy(np.array(x, dtype=np.float32))
        ty.copy_from_numpy(np.array(y, dtype=np.int32))
        #print 'batch size: ', tx.shape, ty.shape
        grads, (l, a) = net.train(tx, ty)
        loss += l
        acc += a
        for (s, p, g) in zip(net.param_names(), net.param_values(), grads):
            opt.apply_with_lr(epoch, get_lr(epoch), g, p, str(s))
        # update progress bar
        utils.update_progress(b * 1.0 / num_train_batch,
                              'training loss = %f, accuracy = %f' % (l, a))
        print 'training time:', time.time() - start
        if b == 10:
            break
    return loss, acc

def test_one_epoch(net, opt, get_lr, epoch, bin_folder, 
         num_test_batch, batch_size, mean, dev): 
    loss, acc = 0.0, 0.0
    tx = tensor.Tensor((batch_size, 3, 224, 224), dev)
    ty = tensor.Tensor((batch_size,), dev, core_pb2.kInt)
    fp = open(bin_folder+'/test.bin', 'rb')
    for b in range(num_test_batch):
        data = load_batch(fp, batch_size)
        pool = Pool(processes=24)
        tmp = pool.map(multi_run_wrapper, izip(repeat(0), data, repeat(mean), repeat(224)))
        pool.close()
        pool.join()
        x, y = zip(*tmp)
        tx.copy_from_numpy(np.array(x, dtype=np.float32))
        ty.copy_from_numpy(np.array(y, dtype=np.float32))
        print np.array(x), y[0]
        l, a = net.evaluate(tx, ty)
        loss += l
        acc += a
    return loss, acc 

def train(bin_folder, net, max_epoch, get_lr, weight_decay, 
          num_train_batch, num_train_files,  num_test_batch, mean, 
          train_batch_size=64, test_batch_size=50, use_cpu=False):
    print 'Start intialization............'
    if use_cpu:
        print 'Using CPU'
        dev = device.get_default_device()
    else:
        print 'Using GPU'
        dev = device.create_cuda_gpu()

    net.to_device(dev)
    opt = optimizer.SGD(momentum=0.9, weight_decay=weight_decay)
    for (p, specs) in zip(net.param_names(), net.param_specs()):
        opt.register(p, specs)
    print "Initialization done."

    for epoch in range(max_epoch):
        print 'Epoch %d' % epoch
        loss, acc = train_one_epoch(net, opt, get_lr, epoch, 
            bin_folder, num_train_batch, num_train_files, train_batch_size, mean, dev)
        info = '\ntraining loss = %f, training accuracy = %f, lr = %f' \
            % (loss / num_train_batch, acc / num_train_batch, get_lr(epoch))
        print info

        loss, acc = test_one_epoch(net, opt, get_lr, epoch, 
            bin_folder, num_test_batch, test_batch_size, mean, dev)
        print 'test loss = %f, test accuracy = %f' \
            % (loss / num_test_batch, acc / num_test_batch)
        break
    net.save('model.bin')  # save model params into checkpoint file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train resnet for imagenet')
    parser.add_argument('model', choices=['resnet'], default='resnet')
    parser.add_argument('data', default='imagenet_data')
    parser.add_argument('--use_cpu', action='store_true')
    args = parser.parse_args()
    assert os.path.exists(args.data), \
        'Pls download the cifar10 dataset via "download_data.py py"'
    print 'Loading data ..................'
    #train_x, train_y = load_train_data(args.data)
    #test_x, test_y = load_test_data(args.data)
    #train_x, test_x = normalize_for_alexnet(train_x, test_x)
    net = resnet.create_net(args.use_cpu)
    mean = load_mean(args.data)
    num_train_image = 12800 #1281167
    num_test_image = 500#00
    train(args.data, net, 200, resnet_lr, 1e-4, num_train_image/32-1, 10,
        num_test_image/10, mean, 32, 50, use_cpu=args.use_cpu)
