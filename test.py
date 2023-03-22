import os
import cv2
import numpy as np
import paddle
import matplotlib.pyplot as plt
from models import UnetGenerator

def test():
    weights_save_path = 'work/weights'

    last_weights_path = os.path.join(weights_save_path, "epoch100.pdparams")
    model_state_dict = paddle.load(last_weights_path)
    generator = UnetGenerator()
    generator.load_dict(model_state_dict)
    generator.eval()
    # 读取数据
    test_names = os.listdir('data/cartoon_A2B/test')
    # img_name = np.random.choice(test_names)
    img_name = '01429.png'
    img_A2B = cv2.imread('data/cartoon_A2B/test/' + img_name)
    img_A = img_A2B[:, 256:]  # 卡通图（即输入）
    # img_B = img_A2B[:, :256]                                  # 真人图（即预测结果）

    # img_A= cv2.imread('data/test4.png')
    # img_A = img_A[:, 256:]

    g_input = img_A.astype('float32') / 127.5 - 1  # 归一化
    g_input = g_input[np.newaxis, ...].transpose(0, 3, 1, 2)  # NHWC -> NCHW
    g_input = paddle.to_tensor(g_input)  # numpy -> tensor

    g_output = generator(g_input)
    g_output = g_output.detach().numpy()  # tensor -> numpy
    g_output = g_output.transpose(0, 2, 3, 1)[0]  # NCHW -> NHWC
    g_output = g_output * 127.5 + 127.5  # 反归一化
    g_output = g_output.astype(np.uint8)

    img_show = np.hstack([img_A, g_output])[:, :, ::-1]
    plt.figure(figsize=(8, 8))
    plt.imshow(img_show)
    plt.show()

if __name__ == '__main__':
    test()
