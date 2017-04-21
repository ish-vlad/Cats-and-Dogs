import resnet50
import lasagne
import theano
import theano.tensor as T
import numpy as np
import os, sys
import skimage.transform
import skimage.io
import pickle

path = 'train/cropped_samples224/'
output_file = 'train/resnet_train_features.csv'

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/resnet50.pkl
model = pickle.load(open('resnet50.pkl', 'rb'))
classes = model['synset_words']
mean_image = model['mean_image']

#computational graph construction
net = resnet50.build_model()

#load weights
lasagne.layers.set_all_param_values(net['prob'], model['values'])

#define placeholder variables
input_var = T.tensor4('input_var')
output_var = lasagne.layers.get_output(net['fc1000'], {net['input']: input_var}, deterministic=True)

#compile prediction function
predict_fn = theano.function([input_var], output_var)

def preprocess(img):
    if len(img.shape) == 2:
        return np.array([np.repeat(np.array(img), 3).reshape(3,224,-1)])
    return np.array([np.swapaxes(np.swapaxes(img, 0,1), 0,2)])

from os import listdir

i = 0
print('Start')
for filename in listdir(path):
    img = skimage.io.imread(path+filename)
    prob = predict_fn(preprocess(img)).ravel()

    with open(output_file, 'a+') as f:
        f.write(filename + ',' + ','.join(prob.astype(str)) + '\n')
    
    i+=1
    if i % 100 == 0:
        print(i)