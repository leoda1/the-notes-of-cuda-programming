import torch
import torch.nn as nn
import torchvision
import torch.onnx
import os 


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 18, 3)
        self.conv2 = torchvision.ops.DeformConv2d(3, 3, 3)
    
    def forward(self, x):
        return self.conv2(x, self.conv1(x))

def infer():
    input = torch.randn(1, 3, 5, 5)

    model = Model()
    x = model(input)
    print("input: ", input.data)
    print("results: ", x.data)

def export_onnx():
    input = torch.randn(1, 3, 5, 5)
    
    model = Model()
    model.eval()

    output_path = "models"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path, "sample-deformable.onnx")
    torch.onnx.export(
        model = model,
        args = (input,),
        f = output_file,
        export_params = True,
        input_names  = ['input'],
        output_names = ['output'],
        opset_version=12,)
    print(f"Model successfully exported to {output_file}") 

if __name__ == '__main__':
    infer()
    export_onnx()
 