
import numpy as np
import tensorrt as trt
import math


logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
builder.max_batch_size = 1
builder.max_workspace_size = 1<<20

network = builder.create_network()

params = np.load('/home/nvidia/procedure/mobilenet_v2_ssdlite_keras-master/mb_weights/mobilenet_segnet.npz')
ENGINE_PATH = '/home/nvidia/procedure/mobilenet_v2_ssdlite_keras-master/mobile_segnet.engine'
#net = network.add_padding(input = , pre_padding = , post_padding = )

input_channels = 3
input_height = 224
input_width = 224 

DTYPE_trt = trt.float32
DTYPE_numpy = np.float32



input_tensor = network.add_input(name="input_1", dtype = DTYPE_trt, shape=[input_channels, input_height, input_width])

##---- add network layers -------


def conv2d_trt(inputs_net, conv_weights, num_kernel, kernel_size, conv_stride, conv_padding):
    #
    w = conv_weights.transpose((3,2,0,1)).reshape(-1)
    b = np.zeros(num_kernel, dtype = DTYPE_numpy)
    #
    net = network.add_convolution(inputs_net, num_kernel, kernel_size, w, b)
    net.stride = conv_stride
    net.padding = conv_padding
    print(net.get_output(0).shape)
    return net


def bn_trt(inputs_net, bn_gamma, bn_beta, bn_mean, bn_var):
    #
    gamma0 = bn_gamma.reshape(-1)
    beta0 = bn_beta.reshape(-1)
    moving_mean0 = bn_mean.reshape(-1)
    moving_var0 = bn_var.reshape(-1)
    #
    scale0 = gamma0 / np.sqrt(moving_var0 + 0.001)
    shift0 = -moving_mean0 / np.sqrt(moving_var0 + 0.001) * gamma0 + beta0
    power0 = np.ones(len(gamma0), dtype = DTYPE_numpy)
    #
    net = network.add_scale(inputs_net.get_output(0), trt.ScaleMode.CHANNEL, shift0, scale0, power0)
    print(net.get_output(0).shape)
    return net


def relu6_trt(inputs_net):
    relu6_net = network.add_activation(inputs_net.get_output(0), trt.ActivationType.RELU)
    max_val = np.full([inputs_net.get_output(0).shape[0], inputs_net.get_output(0).shape[1], inputs_net.get_output(0).shape[2]], 6, dtype = DTYPE_numpy)
    trt_6 = network.add_constant(max_val.shape, max_val)
    net = network.add_elementwise(relu6_net.get_output(0), trt_6.get_output(0), trt.ElementWiseOperation.MIN)
    print(net.get_output(0).shape)
    return net



def depthwise_conv2d_trt(inputs_net, strides, tf_tensors, conv_padding):
    dw_filter = tf_tensors
    dw_filter = np.expand_dims(dw_filter, axis=0)
    dw_filter = np.transpose(dw_filter, (0,4,3,1,2))
    dw_filter = np.ascontiguousarray(dw_filter, dtype = DTYPE_numpy)
    dw_filter_shape = (dw_filter.shape[-1], dw_filter.shape[-1])
    dw_bias = np.zeros(dw_filter.shape[2], dtype = DTYPE_numpy)
    dw_bias = np.ascontiguousarray(dw_bias, dtype = DTYPE_numpy)
    net = network.add_convolution(input=inputs_net.get_output(0), num_output_maps=int(dw_filter.shape[2]), kernel_shape=dw_filter_shape, kernel=dw_filter, bias=dw_bias)
    net.stride = strides
    net.num_groups = dw_filter.shape[2]
    net.padding = conv_padding
    print(net.get_output(0).shape)
    return net

def pointwise_conv2d_trt(inputs_net, conv_weights, num_kernel, kernel_size, conv_stride):
    #
    w = conv_weights.transpose((3,2,0,1)).reshape(-1)
    b = np.zeros(num_kernel, dtype = DTYPE_numpy)
    #
    net = network.add_convolution(inputs_net, num_kernel, kernel_size, w, b)
    net.stride = conv_stride
    print(net.get_output(0).shape)
    return net


def upsampling_trt(inputs_net, scale_factor = 2, resize_mode = trt.ResizeMode.NEAREST):
    input_shape = inputs_net.get_output(0).shape
    net = network.add_resize(inputs_net.get_output(0))
    net.resize_mode = resize_mode
    net.scales = [1, scale_factor, scale_factor]
    net.shape = (input_shape[0], input_shape[1]*scale_factor, input_shape[2]*scale_factor)
    net.align_corners = False
    print(net.get_output(0).shape)
    return net

def reshape_trt(inputs_net):
    net = network.add_shuffle(inputs_net.get_output(0))
    net.first_transpose = (1,2,0)
    h = net.get_output(0).shape[0]
    w = net.get_output(0).shape[1]
    c = net.get_output(0).shape[2]
    net.reshape_dims = (h*w, -1)
    print(net.get_output(0).shape)
    return net

def softmax_trt(inputs_net):
    net = network.add_softmax(inputs_net.get_output(0))
    print(net.get_output(0).shape)
    return net

#conv1
net = conv2d_trt(inputs_net = input_tensor, conv_weights = params['conv1/kernel:0'], num_kernel = 32, kernel_size = (3, 3), conv_stride = (2, 2), conv_padding = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv1_bn/gamma:0'], bn_beta = params['conv1_bn/beta:0'], bn_mean = params['conv1_bn/moving_mean:0'], bn_var = params['conv1_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)

## conv_dw1
net = depthwise_conv2d_trt(inputs_net = net, strides = (1, 1), tf_tensors = params['conv_dw_1/depthwise_kernel:0'], conv_padding = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_dw_1_bn/gamma:0'], bn_beta = params['conv_dw_1_bn/beta:0'], bn_mean = params['conv_dw_1_bn/moving_mean:0'], bn_var = params['conv_dw_1_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)
## conv_pw1
net = pointwise_conv2d_trt(inputs_net = net.get_output(0), conv_weights = params['conv_pw_1/kernel:0'], num_kernel = 64, kernel_size = (1, 1), conv_stride = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_pw_1_bn/gamma:0'], bn_beta = params['conv_pw_1_bn/beta:0'], bn_mean = params['conv_pw_1_bn/moving_mean:0'], bn_var = params['conv_pw_1_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)



## conv_dw2
net = depthwise_conv2d_trt(inputs_net = net, strides = (2, 2), tf_tensors = params['conv_dw_2/depthwise_kernel:0'], conv_padding = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_dw_2_bn/gamma:0'], bn_beta = params['conv_dw_2_bn/beta:0'], bn_mean = params['conv_dw_2_bn/moving_mean:0'], bn_var = params['conv_dw_2_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)
## conv_pw2
net = pointwise_conv2d_trt(inputs_net = net.get_output(0), conv_weights = params['conv_pw_2/kernel:0'], num_kernel = 128, kernel_size = (1, 1), conv_stride = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_pw_2_bn/gamma:0'], bn_beta = params['conv_pw_2_bn/beta:0'], bn_mean = params['conv_pw_2_bn/moving_mean:0'], bn_var = params['conv_pw_2_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)


## conv_dw3
net = depthwise_conv2d_trt(inputs_net = net, strides = (1, 1), tf_tensors = params['conv_dw_3/depthwise_kernel:0'], conv_padding = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_dw_3_bn/gamma:0'], bn_beta = params['conv_dw_3_bn/beta:0'], bn_mean = params['conv_dw_3_bn/moving_mean:0'], bn_var = params['conv_dw_3_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)
## conv_pw3
net = pointwise_conv2d_trt(inputs_net = net.get_output(0), conv_weights = params['conv_pw_3/kernel:0'], num_kernel = 128, kernel_size = (1, 1), conv_stride = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_pw_3_bn/gamma:0'], bn_beta = params['conv_pw_3_bn/beta:0'], bn_mean = params['conv_pw_3_bn/moving_mean:0'], bn_var = params['conv_pw_3_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)


## conv_dw4
net = depthwise_conv2d_trt(inputs_net = net, strides = (2, 2), tf_tensors = params['conv_dw_4/depthwise_kernel:0'], conv_padding = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_dw_4_bn/gamma:0'], bn_beta = params['conv_dw_4_bn/beta:0'], bn_mean = params['conv_dw_4_bn/moving_mean:0'], bn_var = params['conv_dw_4_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)
## conv_pw4
net = pointwise_conv2d_trt(inputs_net = net.get_output(0), conv_weights = params['conv_pw_4/kernel:0'], num_kernel = 256, kernel_size = (1, 1), conv_stride = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_pw_4_bn/gamma:0'], bn_beta = params['conv_pw_4_bn/beta:0'], bn_mean = params['conv_pw_4_bn/moving_mean:0'], bn_var = params['conv_pw_4_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)

## conv_dw5
net = depthwise_conv2d_trt(inputs_net = net, strides = (1, 1), tf_tensors = params['conv_dw_5/depthwise_kernel:0'], conv_padding = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_dw_5_bn/gamma:0'], bn_beta = params['conv_dw_5_bn/beta:0'], bn_mean = params['conv_dw_5_bn/moving_mean:0'], bn_var = params['conv_dw_5_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)
## conv_pw5
net = pointwise_conv2d_trt(inputs_net = net.get_output(0), conv_weights = params['conv_pw_5/kernel:0'], num_kernel = 256, kernel_size = (1, 1), conv_stride = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_pw_5_bn/gamma:0'], bn_beta = params['conv_pw_5_bn/beta:0'], bn_mean = params['conv_pw_5_bn/moving_mean:0'], bn_var = params['conv_pw_5_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)

## conv_dw6
net = depthwise_conv2d_trt(inputs_net = net, strides = (2, 2), tf_tensors = params['conv_dw_6/depthwise_kernel:0'], conv_padding = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_dw_6_bn/gamma:0'], bn_beta = params['conv_dw_6_bn/beta:0'], bn_mean = params['conv_dw_6_bn/moving_mean:0'], bn_var = params['conv_dw_6_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)
## conv_pw6
net = pointwise_conv2d_trt(inputs_net = net.get_output(0), conv_weights = params['conv_pw_6/kernel:0'], num_kernel = 512, kernel_size = (1, 1), conv_stride = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_pw_6_bn/gamma:0'], bn_beta = params['conv_pw_6_bn/beta:0'], bn_mean = params['conv_pw_6_bn/moving_mean:0'], bn_var = params['conv_pw_6_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)

## conv_dw7
net = depthwise_conv2d_trt(inputs_net = net, strides = (1, 1), tf_tensors = params['conv_dw_7/depthwise_kernel:0'], conv_padding = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_dw_7_bn/gamma:0'], bn_beta = params['conv_dw_7_bn/beta:0'], bn_mean = params['conv_dw_7_bn/moving_mean:0'], bn_var = params['conv_dw_7_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)
## conv_pw7
net = pointwise_conv2d_trt(inputs_net = net.get_output(0), conv_weights = params['conv_pw_7/kernel:0'], num_kernel = 512, kernel_size = (1, 1), conv_stride = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_pw_7_bn/gamma:0'], bn_beta = params['conv_pw_7_bn/beta:0'], bn_mean = params['conv_pw_7_bn/moving_mean:0'], bn_var = params['conv_pw_7_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)


## conv_dw8
net = depthwise_conv2d_trt(inputs_net = net, strides = (1, 1), tf_tensors = params['conv_dw_8/depthwise_kernel:0'], conv_padding = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_dw_8_bn/gamma:0'], bn_beta = params['conv_dw_8_bn/beta:0'], bn_mean = params['conv_dw_8_bn/moving_mean:0'], bn_var = params['conv_dw_8_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)
## conv_pw8
net = pointwise_conv2d_trt(inputs_net = net.get_output(0), conv_weights = params['conv_pw_8/kernel:0'], num_kernel = 512, kernel_size = (1, 1), conv_stride = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_pw_8_bn/gamma:0'], bn_beta = params['conv_pw_8_bn/beta:0'], bn_mean = params['conv_pw_8_bn/moving_mean:0'], bn_var = params['conv_pw_8_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)

## conv_dw9
net = depthwise_conv2d_trt(inputs_net = net, strides = (1, 1), tf_tensors = params['conv_dw_9/depthwise_kernel:0'], conv_padding = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_dw_9_bn/gamma:0'], bn_beta = params['conv_dw_9_bn/beta:0'], bn_mean = params['conv_dw_9_bn/moving_mean:0'], bn_var = params['conv_dw_9_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)
## conv_pw9
net = pointwise_conv2d_trt(inputs_net = net.get_output(0), conv_weights = params['conv_pw_9/kernel:0'], num_kernel = 512, kernel_size = (1, 1), conv_stride = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_pw_9_bn/gamma:0'], bn_beta = params['conv_pw_9_bn/beta:0'], bn_mean = params['conv_pw_9_bn/moving_mean:0'], bn_var = params['conv_pw_9_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)

## conv_dw10
net = depthwise_conv2d_trt(inputs_net = net, strides = (1, 1), tf_tensors = params['conv_dw_10/depthwise_kernel:0'], conv_padding = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_dw_10_bn/gamma:0'], bn_beta = params['conv_dw_10_bn/beta:0'], bn_mean = params['conv_dw_10_bn/moving_mean:0'], bn_var = params['conv_dw_10_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)
## conv_pw10
net = pointwise_conv2d_trt(inputs_net = net.get_output(0), conv_weights = params['conv_pw_10/kernel:0'], num_kernel = 512, kernel_size = (1, 1), conv_stride = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_pw_10_bn/gamma:0'], bn_beta = params['conv_pw_10_bn/beta:0'], bn_mean = params['conv_pw_10_bn/moving_mean:0'], bn_var = params['conv_pw_10_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)

## conv_dw11
net = depthwise_conv2d_trt(inputs_net = net, strides = (1, 1), tf_tensors = params['conv_dw_11/depthwise_kernel:0'], conv_padding = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_dw_11_bn/gamma:0'], bn_beta = params['conv_dw_11_bn/beta:0'], bn_mean = params['conv_dw_11_bn/moving_mean:0'], bn_var = params['conv_dw_11_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)
## conv_pw11
net = pointwise_conv2d_trt(inputs_net = net.get_output(0), conv_weights = params['conv_pw_11/kernel:0'], num_kernel = 512, kernel_size = (1, 1), conv_stride = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['conv_pw_11_bn/gamma:0'], bn_beta = params['conv_pw_11_bn/beta:0'], bn_mean = params['conv_pw_11_bn/moving_mean:0'], bn_var = params['conv_pw_11_bn/moving_variance:0'])
net = relu6_trt(inputs_net = net)


## conv2d_1
net = conv2d_trt(inputs_net = net.get_output(0), conv_weights = params['conv2d_1/kernel:0'], num_kernel = 512, kernel_size = (3, 3), conv_stride = (1, 1), conv_padding = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['batch_normalization_1/gamma:0'], bn_beta = params['batch_normalization_1/beta:0'], bn_mean = params['batch_normalization_1/moving_mean:0'], bn_var = params['batch_normalization_1/moving_variance:0'])

## upsampling2d_1
net = upsampling_trt(inputs_net = net)

## conv2d_2
net = conv2d_trt(inputs_net = net.get_output(0), conv_weights = params['conv2d_2/kernel:0'], num_kernel = 256, kernel_size = (3, 3), conv_stride = (1, 1), conv_padding = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['batch_normalization_2/gamma:0'], bn_beta = params['batch_normalization_2/beta:0'], bn_mean = params['batch_normalization_2/moving_mean:0'], bn_var = params['batch_normalization_2/moving_variance:0'])

## upsampling2d_2
net = upsampling_trt(inputs_net = net)

## conv2d_3
net = conv2d_trt(inputs_net = net.get_output(0), conv_weights = params['conv2d_3/kernel:0'], num_kernel = 128, kernel_size = (3, 3), conv_stride = (1, 1), conv_padding = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['batch_normalization_3/gamma:0'], bn_beta = params['batch_normalization_3/beta:0'], bn_mean = params['batch_normalization_3/moving_mean:0'], bn_var = params['batch_normalization_3/moving_variance:0'])

## upsampling2d_3
net = upsampling_trt(inputs_net = net)

## conv2d_4
net = conv2d_trt(inputs_net = net.get_output(0), conv_weights = params['conv2d_4/kernel:0'], num_kernel = 64, kernel_size = (3, 3), conv_stride = (1, 1), conv_padding = (1, 1))
net = bn_trt(inputs_net = net, bn_gamma = params['batch_normalization_4/gamma:0'], bn_beta = params['batch_normalization_4/beta:0'], bn_mean = params['batch_normalization_4/moving_mean:0'], bn_var = params['batch_normalization_4/moving_variance:0'])

## conv2d_5
net = conv2d_trt(inputs_net = net.get_output(0), conv_weights = params['conv2d_5/kernel:0'], num_kernel = 5, kernel_size = (3, 3), conv_stride = (1, 1), conv_padding = (1, 1))

## reshape_1
net = reshape_trt(net)

# activation_1
net = softmax_trt(inputs_net = net)

##---- add network layers -------
network.mark_output(net.get_output(0))
print(net.get_output(0).shape)


with builder.build_cuda_engine(network) as engine:
    serialized_engine = engine.serialize()
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        with open(ENGINE_PATH, "wb") as f:
            f.write(engine.serialize())
            print("ok")



