import torch
import torch.nn as nn
import torch.onnx


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        self.bn1   = nn.BatchNorm2d(num_features=6)
        self.act1  = nn.ReLU()
    def forward(self, x):
        return self.act1(self.bn1(self.conv1(x)))

def export_norm_onnx():
    input = torch.randn(1, 3, 6, 6)
    model = Model()
    model.eval()

    torch.onnx.export(
        model = model,
        args  = (input),
        f     = "model.onnx",
        input_names = ["input"],
        output_names = ["output"],
        opset_version = 15)
    print("Finished exporting normal ONNX model")

if __name__ == '__main__':
    export_norm_onnx()