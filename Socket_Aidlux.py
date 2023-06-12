import socket
import time
import cv2
import numpy
import copy


# aidlux相关
from cvs import *
import aidlite_gpu
from utils import detect_postprocess, preprocess_img, draw_detect_res
#, extract_detect_res

import time
import cv2

def ReceiveVideo():
    # IP地址'0.0.0.0'为等待客户端连接
    address = ('192.168.31.5', 9000)
    # 建立socket对象，参数意义见https://blog.csdn.net/rebelqsp/article/details/22109925
    # socket.AF_INET：服务器之间网络通信
    # socket.SOCK_STREAM：流式socket , for TCP
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 将套接字绑定到地址, 在AF_INET下,以元组（host,port）的形式表示地址.
    s.bind(address)
    # 开始监听TCP传入连接。参数指定在拒绝连接之前，操作系统可以挂起的最大连接数量。该值至少为1，大部分应用程序设为5就可以了。
    s.listen(5)

    def recvall(sock, count):
        buf = b''  # buf是一个byte类型
        while count:
            # 接受TCP套接字的数据。数据以字符串形式返回，count指定要接收的最大数据量.
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    # 接受TCP连接并返回（conn,address）,其中conn是  新的套接字对象，可以用来接收和发送数据。addr是连接客户端的地址。
    # 没有连接则等待有连接
    conn, addr = s.accept()
    print('connect from PC:' + str(addr))
    if 1:
        start = time.time()  # 用于计算帧率信息
        length = recvall(conn, 16)  # 获得图片文件的长度,16代表获取长度
        stringData = recvall(conn, int(length))  # 根据获得的文件长度，获取图片文件
        data = numpy.frombuffer(stringData, numpy.uint8)  # 将获取到的字符流数据转换成1维数组
        decimg = cv2.imdecode(data, cv2.IMREAD_COLOR)  # 将数组解码成图像
        cv2.imwrite("car.jpg",decimg)
        print("save image ")
        # cv2.imshow('SERVER', decimg)  # 显示图像
        # cv2.waitKey(2000)
        #
        # # 进行下一步处理
        # # 。
        # # 。
        # # 。
        #

        # AidLite初始化：调用AidLite进行AI模型的加载与推理，需导入aidlite
        aidlite = aidlite_gpu.aidlite()
        # Aidlite模型路径
        model_path = '/home//models/yolov5n-fp16.tflite'
        # 定义输入输出shape
        in_shape = [1 * 640 * 640 * 3 * 4]
        out_shape = [1 * 25200 * 85 * 4]
        # 加载Aidlite检测模型：支持tflite, tnn, mnn, ms, nb格式的模型加载
        aidlite.ANNModel(model_path, in_shape, out_shape, 4, 0)

        # 读取图片进行推理
        # 设置测试集路径
        source = "/home/AIGC/images/AIGC"
        images_list = os.listdir(source)
        print(images_list)
        frame_id = 0
        # 读取数据集
        for image_name in images_list:
            frame_id += 1
            print("frame_id:", frame_id)
            image_path = os.path.join(source, image_name)
            frame = cvs.imread(image_path)

            # 预处理
            img = preprocess_img(frame, target_shape=(640, 640), div_num=255, means=None, stds=None)
            # 数据转换：因为setTensor_Fp32()需要的是float32类型的数据，所以送入的input的数据需为float32,大多数的开发者都会忘记将图像的数据类型转换为float32
            aidlite.setInput_Float32(img, 640, 640)
            # 模型推理API
            aidlite.invoke()
            # 读取返回的结果
            pred = aidlite.getOutput_Float32(0)
            # 数据维度转换
            pred = pred.reshape(1, 25200, 85)[0]
            # 模型推理后处理
            pred = detect_postprocess(pred, frame.shape, [640, 640, 3], conf_thres=0.25, iou_thres=0.45)
            # 绘制推理结果
            res_img = draw_detect_res(frame, pred)
            cvs.imshow(res_img)

            # 测试结果展示停顿
            time.sleep(5)




        # # 将帧率信息回传，主要目的是测试可以双向通信
        end = time.time()
        seconds = end - start
        fps = 1 / seconds
        ##返回已处理图像到客户端
        conn.send(bytes(str(int(fps)), encoding='utf-8'))
        # image = copy.deepcopy(decimg)
        # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        # result, imgencode = cv2.imencode('.jpg', image, encode_param)
        # # 建立矩阵
        # data = numpy.array(imgencode)
        # # 将numpy矩阵转换成字符形式，以便在网络中传输
        # img_Data = data.tostring()

        # # 先发送要发送的数据的长度
        # # ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串
        # conn.send(str.encode(str(len(img_Data)).ljust(16)))
        # # # print(img_Data)
        # # # 发送数据
        # conn.send(img_Data)

        # if cv2.waitKey(10) & 0xff == 27:
        #     break
    s.close()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    ReceiveVideo()

