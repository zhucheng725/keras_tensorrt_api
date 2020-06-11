# keras_tensorrt_api
keras_tensorrt_api
img = cv2.imread("001.jpg")

img_ = img[:,:,::-1].transpose((2,0,1))

① 在opencv里，图格式HWC，其余都是CHW，故transpose((2,0,1))

② img[:,:,::-1]对应H、W、C，彩图是3通道，即C是3层。opencv里对应BGR，故通过C通道的 ::-1 就是把BGR转为RGB

 注：  [::-1] 代表顺序相反操作

③ 若不涉及C通道的BGR转RGB，如Img[:,:,0]代表B通道，也就是蓝色分量图像；Img[:,:,1]代表G通道，也就是绿色分量图像；

     Img[:,:,2]代表R通道，也就是红色分量图像。
