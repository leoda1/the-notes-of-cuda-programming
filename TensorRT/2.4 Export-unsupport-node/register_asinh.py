import torch
import torch.onnx
import onnxruntime
import os
from torch.onnx import register_custom_op_symbolic
#===========================================================================
#                                  方案一
#===========================================================================
# 创建asinh算子的symblic来实现‘登记’
# 调用g.op()函数来创建算子节点，来将Asinh算子注册到ONNX的符号注册表中
# g就是graph的缩写，表示计算图
# symbolic的参数与pytorch的Asinh接口函数的参数对齐即可
#   def asin(input: Tensor, *, out: Optional[Tensor]=None) -> Tensor: ...
# def asinh_symbolic(g, input, *, out=None):
#     return g.op("Asinh", input) #Asinh算子的ONNX算子名，在Operators.md中对应

# # 注册asinh_symbolic这个符号函数 与PyTorch中的Asinh函数绑定
# register_custom_op_symbolic('aten::asinh', asinh_symbolic, 9)

#===========================================================================
#                                  方案二
#===========================================================================
# pytorch官方写法
import functools
from torch.onnx._internal import _beartype, jit_utils, registration
_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=9)
@_onnx_symbolic("aten::asinh")
@_beartype.beartype
def asinh(g: jit_utils.GraphContext, self):
    return g.op("Asinh", self)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.asinh(x)
    
def infer():
    input = torch.rand(1, 5)
    #======================
    # PyTorch inference
    #======================
    model = Model()
    x = model(input)
    print("Pytorch inference result:", x)
    #======================
    # ONNX inference
    #======================
    sess = onnxruntime.InferenceSession("models/sample-asinh.onnx")
    x = sess.run(None, {'input': input.numpy()})
    print("ONNX inference result:", x)

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

if __name__ == '__main__':
    export_onnx()
    infer()