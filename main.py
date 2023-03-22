import os
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from tqdm import tqdm
from models import UnetGenerator, NLayerDiscriminator
from utils.datasets import PairedData

import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt



paired_dataset_train = PairedData('train')
paired_dataset_test = PairedData('test')


# ... 定义超参数、优化器、损失函数和数据加载器
# ... 将训练循环放在main函数中
generator = UnetGenerator()
discriminator = NLayerDiscriminator()

out = generator(paddle.ones([1, 3, 256, 256]))
print('生成器输出尺寸：', out.shape)

out = discriminator(paddle.ones([1, 6, 256, 256]))
print('鉴别器输出尺寸：', out.shape)


# 超参数
LR = 1e-4
BATCH_SIZE = 8
EPOCHS = 100

# 优化器
optimizerG = paddle.optimizer.Adam(
    learning_rate=LR,
    parameters=generator.parameters(),
    beta1=0.5,
    beta2=0.999)

optimizerD = paddle.optimizer.Adam(
    learning_rate=LR,
    parameters=discriminator.parameters(),
    beta1=0.5,
    beta2=0.999)

# 损失函数
bce_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

# dataloader
data_loader_train = DataLoader(
    paired_dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
    )

data_loader_test = DataLoader(
    paired_dataset_test,
    batch_size=BATCH_SIZE
    )


def main():
    # 训练数据统计
    train_names = os.listdir('data/cartoon_A2B/train')
    print(f'训练集数据量: {len(train_names)}')

    # 测试数据统计
    test_names = os.listdir('data/cartoon_A2B/test')
    print(f'测试集数据量: {len(test_names)}')

    # 训练数据可视化
    imgs = []
    for img_name in np.random.choice(train_names, 3, replace=False):
        imgs.append(cv2.imread('data/cartoon_A2B/train/' + img_name))

    plt.imshow(imgs[0])
    plt.imshow(imgs[1])
    plt.imshow(imgs[2])
    img_show = np.vstack(imgs)[:, :, ::-1]
    plt.figure(figsize=(10, 10))
    plt.imshow(img_show)
    plt.show()

    results_save_path = 'work/results'
    os.makedirs(results_save_path, exist_ok=True)  # 保存每个epoch的测试结果

    weights_save_path = 'work/weights'
    os.makedirs(weights_save_path, exist_ok=True)  # 保存模型

    for epoch in range(EPOCHS):
        for data in tqdm(data_loader_train):
            real_A, real_B = data

            optimizerD.clear_grad()
            # D(real)
            real_AB = paddle.concat((real_A, real_B), 1)
            d_real_predict = discriminator(real_AB)
            d_real_loss = bce_loss(d_real_predict, paddle.ones_like(d_real_predict))

            # D(fake)
            fake_B = generator(real_A).detach()
            fake_AB = paddle.concat((real_A, fake_B), 1)
            d_fake_predict = discriminator(fake_AB)
            d_fake_loss = bce_loss(d_fake_predict, paddle.zeros_like(d_fake_predict))

            # train D
            d_loss = (d_real_loss + d_fake_loss) / 2.
            d_loss.backward()
            optimizerD.step()

            optimizerG.clear_grad()
            # D(fake)
            fake_B = generator(real_A)
            fake_AB = paddle.concat((real_A, fake_B), 1)
            g_fake_predict = discriminator(fake_AB)
            g_bce_loss = bce_loss(g_fake_predict, paddle.ones_like(g_fake_predict))
            g_l1_loss = l1_loss(fake_B, real_B) * 100.
            g_loss = g_bce_loss + g_l1_loss

            # train G
            g_loss.backward()
            optimizerG.step()

        print(f'Epoch [{epoch + 1}/{EPOCHS}] Loss D: {d_loss.numpy()}, Loss G: {g_loss.numpy()}')

        if (epoch + 1) % 10 == 0:
            paddle.save(generator.state_dict(),
                        os.path.join(weights_save_path, 'epoch' + str(epoch + 1).zfill(3) + '.pdparams'))

            # test
            generator.eval()
            with paddle.no_grad():
                for data in data_loader_test:
                    real_A, real_B = data
                    break

                fake_B = generator(real_A)
                result = paddle.concat([real_A[:3], real_B[:3], fake_B[:3]], 3)

                result = result.detach().numpy().transpose(0, 2, 3, 1)
                result = np.vstack(result)
                result = (result * 127.5 + 127.5).astype(np.uint8)

            cv2.imwrite(os.path.join(results_save_path, 'epoch' + str(epoch + 1).zfill(3) + '.png'), result)

            generator.train()

if __name__ == '__main__':
    main()
