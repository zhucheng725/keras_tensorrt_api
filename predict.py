

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time

engine_file = '/home/nvidia/procedure/mobilenet_v2_ssdlite_keras-master/mobile_segnet.engine'
Input_shape = (224, 224, 3)
DTYPE = trt.float32
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')


VOC_COLOR = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]], dtype=np.uint8)

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']



def allocate_buffers(engine):
    print('allocate buffers')
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype= trt.nptype(DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype= trt.nptype(DTYPE))		
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    return h_input, d_input, h_output, d_output
		

def do_inference(context, h_input, d_input, h_output, d_output):
    #transfer input data to the GPU
    cuda.memcpy_htod(d_input, h_input)
    #run inference
    context.execute(batch_size = 1, bindings=[int(d_input), int(d_output)])
    #transfer predictions back from GPU
    cuda.memcpy_dtoh(h_output, d_output)
    return h_output


def load_input(img_path, host_buffer):
    h, w, c = Input_shape
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (w, h))
    img = img.astype(np.float32)
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = img[:, :, ::-1]
    img = img.transpose((2,0,1))# for tensorrt api rebulid
    #img = img.reshape((c,w,h))
    img = img.reshape(-1)
    np.copyto(host_buffer, img)

def write_img(output, k):
    a = output
    a = a.reshape((112,112,5))
    pre = np.zeros((112,112,3))
    for i in range(112):
        for j in range(112):
            result_color = np.argmax(a[i,j,:])
            #if result_color == 7 or result_color == 15:
            pre[i,j] =  VOC_COLOR[result_color]      
    img =  cv2.imread('/home/nvidia/procedure/keras/model/1/image_' + str(k)+'.jpg')
    pre =  cv2.resize(pre,(1280,720), interpolation =  cv2.INTER_AREA)
    pre = pre.astype(np.float32)
    img = img.astype(np.float32)
    overlapping = cv2.addWeighted(img, 0.7, pre, 1.0, 0)
    cv2.imwrite('/home/nvidia/procedure/keras/model/1_result/image_'+ str(k)+'.jpg',overlapping)


def write_label(output):
    a = output
    a = a.reshape((112,112,5))
    pre = np.zeros((112,112))
    for i in range(112):
        for j in range(112):
            pre[i,j] = np.argmax(a[i,j,:])
 

def write_img_argmax(output, k):
    a = output
    a = a.reshape((112,112))
    print('np.unique(a)',np.unique(a))
    pre = np.zeros((112,112,3))
    for i in range(112):
        for j in range(112):
            #if result_color == 7 or result_color == 15:
            pre[i,j] =  VOC_COLOR[int(a[i,j])]      
    #img =  cv2.imread('/home/nvidia/procedure/keras/model/1/image_' + str(k)+'.jpg')
    #pre =  cv2.resize(pre,(1280,720), interpolation =  cv2.INTER_AREA)
    #pre = pre.astype(np.float32)
    #img = img.astype(np.float32)
    #overlapping = cv2.addWeighted(img, 0.7, pre, 1.0, 0)
    #cv2.imwrite('/home/nvidia/procedure/keras/model/1_result/image_'+ str(k)+'.jpg',overlapping)


with open(engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())


h_input, d_input, h_output, d_output = allocate_buffers(engine)

with engine.create_execution_context() as context:
    start_time = time.time()
    for i in range(11):
        img_path = '/home/nvidia/procedure/keras/model/1/image_' + str(i) + '.jpg'
        load_input(img_path, host_buffer = h_input)
        output = do_inference(context, h_input, d_input, h_output, d_output)
        #write_img(output, i)
        #write_label(output)
        write_img(output, i)
        print(i)
    end_time = time.time()
    print('use time:', (end_time - start_time))
    print('output.shape',output.shape)



