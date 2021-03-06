import numpy as np
import theano
import theano.tensor as T
import lasagne
import cPickle as pickle
import os
import scipy
from scipy.misc import imread, imsave, imresize
from lasagne.utils import floatX
import sys
import skimage.transform
import skimage.io

path = 'train/cropped_samples224/'

output_file = 'train/vgg_train_features.csv'

# copyright: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.nonlinearities import softmax

def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net

net = build_model()
with open('vgg16.pkl') as f:
    weights = pickle.load(f)
lasagne.layers.set_all_param_values(net["prob"],weights['param values'])



input_image = T.tensor4('input')
output = lasagne.layers.get_output(net['fc7'], input_image,deterministic=True)
predict_fn = theano.function([input_image], output) 

def preprocess(img):
    if len(img.shape) == 2:
        return np.array([np.repeat(np.array(img), 3).reshape(3,224,-1)])
    return np.array([np.swapaxes(np.swapaxes(img, 0,1), 0,2)])

from os import listdir

i = 0
print('Start')
for filename in listdir(path):
    if filename[0] == '.':
        continue
    img = skimage.io.imread(path+filename)
    prob = predict_fn(preprocess(img)).ravel()

    with open(output_file, 'a+') as f:
        f.write(filename + ',' + ','.join(prob.astype(str)) + '\n')
    
    i+=1
    if i % 100 == 0:
        print i,
        sys.stdout.flush()