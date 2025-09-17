import torch
import torch.nn as nn

class AODNet(nn.Module):
    def __init__(self):
        super(AODNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, padding=1)
        self.b = 1

        self.lrelu = nn.LeakyReLU()  # ReLU to LReLU, avoid dead cell

    def forward(self, x):
        x = x * 0.5 + 0.5  # to [0, 1]

        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = self.lrelu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = self.lrelu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = self.conv5(cat3)

        output = k * x - k + self.b
        output = output * 2 - 1  # to [-1, 1]
        return output

if __name__ == "__main__":
    from pyzjr.nn import summary_1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AODNet()
    model = model.to(device)
    inputs = torch.randn(1, 3, 256, 256).to(device)
    output = model(inputs)
    print("output", output.shape)
    summary_1(model, input_size=(3, 256, 256))
    """
    Total params: 1,761
    Trainable params: 1,761
    Non-trainable params: 0

    Input size (MB): 0.750000
    Forward/backward pass size (MB): 13.500000
    Params size (MB): 0.006718
    Estimated Total Size (MB): 14.256718
    """
