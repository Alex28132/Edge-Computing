import socket
import cv2
import numpy as np
import time
import sys

### 本代码主要是客户端代码，aidlux上的Socket_fuwuduan.py是匹配的服务端代码，当服务端代码启动时，由本代码读取一张图片，推送过去

import torch
# from torch import autocast
from torch.cuda.amp import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler,DDIMScheduler


def recvall(sock, count):
    buf = b''  # buf是一个byte类型
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def SendAIGC():
    # 建立sock连接
    # address要连接的aidlux服务器IP地址和端口号
    address = ('192.168.31.5', 9529)
    try:
        # 建立socket对象
        # socket.AF_INET：服务器之间网络通信
        # socket.SOCK_STREAM：流式socket , for TCP
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 开启连接
        sock.connect(address)
    except socket.error as msg:
        print(msg)
        sys.exit(1)

    ###########传送AIGC图片#################
    # ## 如果本地没有GPU
    # if 1:
    #     frame = cv2.imread("car.png")
    #     # # 压缩参数，后面cv2.imencode将会用到，对于jpeg来说，15代表图像质量，越高代表图像质量越好为 0-100，默认95
    #     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    #     # cv2.imencode将图片格式转换(编码)成流数据，赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输
    #     # '.jpg'表示将图片按照jpg格式编码。
    #     result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    #     # 建立矩阵
    #     data = np.array(imgencode)
    #     # 将numpy矩阵转换成字符形式，以便在网络中传输
    #     stringData = data.tostring()
    #
    #     # 先发送要发送的数据的长度
    #     # ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串
    #     sock.send(str.encode(str(len(stringData)).ljust(16)))
    #     # 发送数据
    #     sock.send(stringData)
    ### 如果本地有GPU
    # if 0:
        ### 本地生成AIGC图片 ###
        ## 添加AIGC代码 ##
        #####################

    model_id = "E:/pycharmwenjian/Aidlux课程/AIGC训练营_Lesson2_code/models/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    # In[13]:

    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    # In[18]:

    prompt = "a car"
    negative_prompt = "(EasyNegative), bad_prompt_version2, (watermark), (signature), (sketch by bad-artist), (signature), (worst quality), (low quality), ((badhandsv5-neg)), ((badhandv4)), (bad anatomy), deformed hands, NSFW, nude, EasyNegative, (worst quality:1.4), (low quality:1.4), (normal quality:1.4),lowres,crowd"
# with autocast("cuda"):

    image = pipe(prompt,
                 negative_prompt=negative_prompt,
                 num_inference_steps=25,
                 width=512,
                 height=768,
                 guidance_scale=7.5).images[0]

    # In[19]:

    image.save("./car.png")



    frame = cv2.imread("./car.png")
    # # 压缩参数，后面cv2.imencode将会用到，对于jpeg来说，15代表图像质量，越高代表图像质量越好为 0-100，默认95
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    # cv2.imencode将图片格式转换(编码)成流数据，赋值到内存缓存中;主要用于图像数据格式的压缩，方便网络传输
    # '.jpg'表示将图片按照jpg格式编码。
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    # 建立矩阵
    data = np.array(imgencode)
    # 将numpy矩阵转换成字符形式，以便在网络中传输
    stringData = data.tostring()

    # 先发送要发送的数据的长度
    # ljust() 方法返回一个原字符串左对齐,并使用空格填充至指定长度的新字符串
    sock.send(str.encode(str(len(stringData)).ljust(16)))
    # 发送数据
    sock.send(stringData)

    # 读取服务器返回值
    receive = sock.recv(16)
    if len(receive):
         print("图片发送成功")
         print(str(receive, encoding='utf-8'))  ### 之前接受的帧率数据，现在换成image流数据
    sock.close()

if __name__ == '__main__':
    SendAIGC()