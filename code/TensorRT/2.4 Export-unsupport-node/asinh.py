import torch
import torch.onnx
import os

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.asinh(x)
def infer():
    input = torch.rand(1, 5)
    model = Model()
    x = model(input)
    print("inpput_size: ", input.size())
    print("output_size: ", x.size())

def export_onnx():
    input = torch.rand(1, 5)
    model = Model()
    model.eval()
    output_path = "models"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path, "sample-asinh.onnx")
    # Export the model to ONNX
    torch.onnx.export(
        model = model,
        args = (input,),
        f = output_file,
        export_params = True,
        input_names  = ['input'],
        output_names = ['output'],
        opset_version=9,)
    print(f"Model successfully exported to {output_file}")

#==================================================================
# 运行infer（）不会报错 但是运行export_onnx()会报错如下：
# RuntimeError: Exporting the operator asinh to ONNX opset version 9 is not supported. Please open a bug to request ONNX export support for the missing operator.
# 原因：asinh()函数在ONNX中没有对应的算子，因此无法导出到ONNX。
#==================================================================

if __name__ == '__main__':
    infer()
    # export_onnx()
    
# 去torch/onnx/symbolic_opset9.py文件中添加asinh()函数的导出代码即可。
    
