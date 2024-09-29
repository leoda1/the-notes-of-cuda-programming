import torch
import torch.onnx
import onnxruntime
import os

OperatorExportTypes = torch._C._onnx.OperatorExportTypes
class CustomOp(torch.autograd.Function):
    @staticmethod # 装饰器，用于将类中定义的方法声明为静态方法。
    def forward(ctx, x: torch.Tensor) -> torch.Tensor: 
        ctx.save_for_backward(x) # ctx 是 torch.autograd.Function 中的一个参数，它代表了上下文（context）对象
        x = x.clamp(min = 0)
        return x / (1 + torch.exp(-x)) 
CustomOp = CustomOp.apply # 装饰器，用于将自定义的 autograd 函数注册到 PyTorch 的 autograd 系统中。

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return CustomOp(x)

def infer():
    input = torch.rand(1, 50).uniform_(-1, 1).reshape(1, 2, 5, 5)
    #======================
    # PyTorch inference
    #======================
    model = Model()
    x = model(input)
    print("Pytorch inference result:", x)
    #======================
    # ONNX inference
    #======================
    sess = onnxruntime.InferenceSession("models/sample-CustomOp.onnx")
    x = sess.run(None, {'input': input.numpy()})
    print("ONNX inference result:", x)

def export_onnx():
    input = torch.rand(1, 50).uniform_(-1, 1).reshape(1, 2, 5, 5)
    model = Model()
    model.eval()

    output_path = "models"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path, "sample-CustomOp.onnx")
    
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

if __name__ == '__main__':
    export_onnx()
    infer()