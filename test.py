import os
import cv2
import numpy as np
import paddle
import matplotlib.pyplot as plt
from models import UnetGenerator

def test(image_path):
    # 加载模型权重
    weights_save_path = 'work/weights'
    last_weights_path = os.path.join(weights_save_path, "epoch100.pdparams")
    model_state_dict = paddle.load(last_weights_path)

    # 加载模型
    generator = UnetGenerator()
    generator.load_dict(model_state_dict)
    generator.eval()

    # 读取数据
    img_A = cv2.imread(image_path)
    img_A = cv2.resize(img_A, (256, 256))
    g_input = img_A.astype('float32') / 127.5 - 1  # 归一化
    g_input = g_input[np.newaxis, ...].transpose(0, 3, 1, 2)  # NHWC -> NCHW
    g_input = paddle.to_tensor(g_input)  # numpy -> tensor

    # 预测输出
    g_output = generator(g_input)
    g_output = g_output.detach().numpy()  # tensor -> numpy
    g_output = g_output.transpose(0, 2, 3, 1)[0]  # NCHW -> NHWC
    g_output = g_output * 127.5 + 127.5  # 反归一化
    g_output = g_output.astype(np.uint8)

    # 显示结果
    img_show = np.hstack([img_A, g_output])[:, :, ::-1]
    plt.figure(figsize=(8, 8))
    plt.imshow(img_show)
    plt.show()

if __name__ == '__main__':
    # 指定输入图像的路径
    image_path = 'data/Cartoon_A2B/test_A/01425A.png'
    test(image_path)


