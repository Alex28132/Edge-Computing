# Edge-Computing

AIGC与AidLux互联应用——AidLux端AIGC测评 AidLux端AIGC测评系统搭建（Aidlux s855边缘计算盒子，安卓手机也可）及拓展研究方向




![e44decffd6284376247032161e1650ba_37f4b923dd8c44a2a4fb22efc6901f8b](https://github.com/Alex28132/Edge-Computing/assets/53650857/002e6186-efef-45a9-a623-4c39a8db2d08)


简单来说，就是pc端通过文生图、图生图等形式生成图片，通过socket传递给Aidlux端，Aidlux进行测评并返回结果。
下面简要介绍下Aidlux：Aidlux是一套基于安卓平台边缘计算系统，类似于在安卓平台中嵌入ubuntu系统，可以在安卓手机中下载，也可以在Aidlux发布的边缘计算盒子中运行，我是在Aidlux s855边缘计算盒子中使用，用安卓手机也可以运行demo，只不过边缘计算盒子考虑工业性，散热、接口更丰富。


本文采用的是本地gpu运行（win系统下运行，ubuntu系统下也可）

在Aidlux端使用：python Socket_Aidlux.py 调用

在pc端使用：python Socket_pc.py 调用

结果：
![771f8bc72a080387391b25dde0fa7ab8_d5a5760e0c6b449f963295edffbfd47f](https://github.com/Alex28132/Edge-Computing/assets/53650857/46a00977-c5d6-4940-a84b-8e9e399def87)
