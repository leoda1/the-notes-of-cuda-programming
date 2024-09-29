import torch
import torch.nn as nn
import torchvision
import torch.onnx
import onnxruntime
import os 
from torch.onnx.symbolic_helper import parse_args
from torch.onnx import register_custom_op_symbolic

@parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i","i", "i", "i", "none")
def dcn_symbolic(
        g,
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        n_weight_grps,
        n_offset_grps,
        use_mask):
    return g.op("custom::deform_conv2d", input, offset)

register_custom_op_symbolic("torchvision::deform_conv2d", dcn_symbolic, 12)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 18, 3)
        self.conv2 = torchvision.ops.DeformConv2d(3, 3, 3)
    
    def forward(self, x):
        return self.conv2(x, self.conv1(x))

def infer():
    input = torch.rand(1, 3, 5, 5)
    #======================
    # PyTorch inference
    #======================
    model = Model()
    x = model(input)
    print("Pytorch inference result:", x)
    #======================
    # ONNX inference
    #======================
    sess = onnxruntime.InferenceSession("models/sample-deformable2.onnx")
    x = sess.run(None, {'input': input.numpy()})
    print("ONNX inference result:", x)

def export_onnx():
    input = torch.rand(1, 3, 5, 5)
    model = Model()
    model.eval()

    output_path = "models"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path, "sample-deformable2.onnx")
    
    # Export the model to ONNX
    torch.onnx.export(
        model = model,
        args = (input,),
        f = output_file,
        export_params = True,
        input_names  = ['input'],
        output_names = ['output'],
        opset_version= 12,)
    print(f"Model successfully exported to {output_file}")

if __name__ == '__main__':
    export_onnx()
    infer()
 