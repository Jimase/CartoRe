import torch
import torch.nn as nn


class UnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(UnetGenerator, self).__init__()

        self.down1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        self.down2 = Downsample(ngf, ngf * 2)
        self.down3 = Downsample(ngf * 2, ngf * 4)
        self.down4 = Downsample(ngf * 4, ngf * 8)
        self.down5 = Downsample(ngf * 8, ngf * 8)
        self.down6 = Downsample(ngf * 8, ngf * 8)
        self.down7 = Downsample(ngf * 8, ngf * 8)

        self.center = Downsample(ngf * 8, ngf * 8)

        self.up7 = Upsample(ngf * 8, ngf * 8, use_dropout=True)
        self.up6 = Upsample(ngf * 8 * 2, ngf * 8, use_dropout=True)
        self.up5 = Upsample(ngf * 8 * 2, ngf * 8, use_dropout=True)
        self.up4 = Upsample(ngf * 8 * 2, ngf * 8)
        self.up3 = Upsample(ngf * 8 * 2, ngf * 4)
        self.up2 = Upsample(ngf * 4 * 2, ngf * 2)
        self.up1 = Upsample(ngf * 2 * 2, ngf)

        self.output_block = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        c = self.center(d7)

        x = self.up7(c, d7)
        x = self.up6(x, d6)
        x = self.up5(x, d5)
        x = self.up4(x, d4)
        x = self.up3(x, d3)
        x = self.up2(x, d2)
        x = self.up1(x, d1)

        x = self.output_block(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=4, stride=2, padding=1):
        super(Downsample, self).__init__()

        self.layers = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=4, stride=2, padding=1, use_dropout=False):
        super(Upsample, self).__init__()

        sequence = [
            nn.ReLU(),
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),            nn.BatchNorm2d(out_dim)
        ]

        if use_dropout:
            sequence.append(nn.Dropout(p=0.5))

        self.layers = nn.Sequential(*sequence)

    def forward(self, x, skip):
        x = self.layers(x)
        x = torch.cat([x, skip], dim=1)
        return x


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=6, ndf=64):
        super(NLayerDiscriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            ConvBlock(ndf, ndf*2),
            ConvBlock(ndf*2, ndf*4),
            ConvBlock(ndf*4, ndf*8, stride=1),

            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.layers(input)

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=4, stride=2, padding=1):
        super(ConvBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import pydot
    from torchsummary import summary

    # 创建模型实例
    model = UnetGenerator()

    # 定义输入数据的形状
    input_shape = (1, 3, 256, 256)  # 假设输入大小为256x256

    # 调用paddle.summary生成模型的摘要信息
    summary(model, input_shape)

    # 获取down1模块的结构
    down1_summary = summary[model]['down1']

    # 创建图形对象
    graph = pydot.Dot(graph_type='digraph')

    # 添加down1模块节点
    down1_node = pydot.Node('down1', label='down1', shape='box')
    graph.add_node(down1_node)

    # 添加down1模块的子节点
    for layer_name in down1_summary:
        layer_node = pydot.Node(layer_name, label=layer_name, shape='box')
        graph.add_node(layer_node)
        graph.add_edge(pydot.Edge('down1', layer_name))

    # 绘制图形
    graph.write_png('down1_module.png')
