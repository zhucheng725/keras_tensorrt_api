
##restore .h5 and to .npz

from keras.models import load_model
import cv2
import time
import numpy as np 
import tensorflow as tf
import keras.backend as K
from keras.backend.tensorflow_backend import set_session

import uff
import graphsurgeon as gs
import tensorrt as trt
from keras.utils.generic_utils import CustomObjectScope
import keras


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.log_device_placement = True
sess =  tf.Session(config = config)
set_session(sess)
keras.backend.get_session().run(tf.initialize_all_variables())



K.set_learning_phase(0)


with CustomObjectScope({'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D, 'relu6':tf.nn.relu6, 'tf':tf}):
    model = load_model('/home/nvidia/procedure/keras/output/mobilenet_segnet.h5')

model.summary()

tf_args = {}

#for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #print('tensor_name',i)



for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    print('tensor_name',i)
    tf_args[i.name] = sess.run(i)

np.savez('/home/nvidia/procedure/mobilenet_v2_ssdlite_keras-master/mb_weights/mobilenet_segnet.npz', **tf_args)


