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
"""The resnet model is adapted from http://torch.ch/blog/2016/02/04/resnets.html
The best validation accuracy we achieved is about 83% without data augmentation.
The performance could be improved by tuning some hyper-parameters, including
learning rate, weight decay, max_epoch, parameter initialization, etc.
"""

import cPickle as pickle

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))
# use the python modules by installing py singa in build/python
# pip install -e .

from singa import layer
from singa import initializer
from singa import metric
from singa import loss
from singa import net as ffnet


def Block(net, name, nb_filters, stride):
    split = net.add(layer.Split(name + "-split", 2))
    if stride > 1:
        net.add(layer.Conv2D(name + "-br2-conv", nb_filters, 1, stride, pad=0), split)
        br2bn = net.add(layer.BatchNormalization(name + "-br2-bn"))
    net.add(layer.Conv2D(name + "-br1-conv1", nb_filters, 3, stride, pad=1), split)
    net.add(layer.BatchNormalization(name + "-br1-bn1"))
    net.add(layer.Activation(name + "-br1-relu"))
    net.add(layer.Conv2D(name + "-br1-conv2", nb_filters, 3, 1, pad=1))
    br1bn2 = net.add(layer.BatchNormalization(name + "-br1-bn2"))
    if stride > 1:
        net.add(layer.Merge(name + "-merge"), [br1bn2, br2bn])
    else:
        net.add(layer.Merge(name + "-merge"), [br1bn2, split])

def Block2(net, name, nb_filters, stride):
    '''
    one building block with two branches:
    branch one contains identity mapping,
    branch two uses a three layer block with 1*1, 3*3, 1*1 filters, respectively.
    '''

    split = net.add(layer.Split(name + "-split", 2))
    if stride > 1:
        net.add(layer.Conv2D(name + "-br2-conv", nb_filters * 4, 1, stride, pad=0), split)
        br2bn = net.add(layer.BatchNormalization(name + "-br2-bn"))
    net.add(layer.Conv2D(name + "-br1-conv1", nb_filters, 1, stride, pad=0), split)
    net.add(layer.BatchNormalization(name + "-br1-bn1"))
    net.add(layer.Activation(name + "-br1-relu1"))
    net.add(layer.Conv2D(name + "-br1-conv2", nb_filters, 3, 1, pad=1))
    net.add(layer.BatchNormalization(name + "-br1-bn2"))
    net.add(layer.Activation(name + "-br1-relu2"))
    net.add(layer.Conv2D(name + "-br1-conv3", nb_filters * 4, 1, 1, pad=0))
    br1bn3 = net.add(layer.BatchNormalization(name + "-br1-bn3"))
    if stride > 1:
        net.add(layer.Merge(name + "-merge"), [br1bn3, br2bn])
    else:
        net.add(layer.Merge(name + "-merge"), [br1bn3, split])
    net.add(layer.Activation(name + "-relu"))

def create_net(use_cpu=False):
    if use_cpu:
        layer.engine = 'singacpp'

    # resnet-50
    net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())
    net.add(layer.Conv2D("conv1", 64, 7, 2, pad=3, input_sample_shape=(3, 224, 224)))
    net.add(layer.BatchNormalization("bn1"))
    net.add(layer.Activation("relu1"))
    net.add(layer.MaxPooling2D("pool1", 3, 2, border_mode='valid'))

    Block2(net, "2a", 64, 1)
    Block2(net, "2b", 64, 1)
    Block2(net, "2c", 64, 1)

    Block2(net, "3a", 128, 2)
    Block2(net, "3b", 128, 1)
    Block2(net, "3c", 128, 1)
    Block2(net, "3d", 128, 1)

    Block2(net, "4a", 256, 2)
    Block2(net, "4b", 256, 1)
    Block2(net, "4c", 256, 1)
    Block2(net, "4d", 256, 1)
    Block2(net, "4e", 256, 1)
    Block2(net, "4f", 256, 1)

    Block2(net, "5a", 512, 2)
    Block2(net, "5b", 512, 1)
    Block2(net, "5c", 512, 1)

    net.add(layer.AvgPooling2D("pool5", 7, 7, border_mode='valid'))
    net.add(layer.Flatten('flat'))
    net.add(layer.Dense('ip5', 1000))
    print 'Start intialization............'
    for (p, name) in zip(net.param_values(), net.param_names()):
        # print name, p.shape
        if 'mean' in name or 'beta' in name:
            p.set_value(0.0)
        elif 'var' in name:
            p.set_value(1.0)
        elif 'gamma' in name:
            initializer.uniform(p, 0, 1)
        elif len(p.shape) > 1:
            if 'conv' in name:
                initializer.gaussian(p, 0, p.shape[1])
                # initializer.gaussian(p, 0, 9.0 * p.shape[0])
            else:
                initializer.uniform(p, p.shape[0], p.shape[1])
        else:
            p.set_value(0)
        # print name, p.l1()

    return net
