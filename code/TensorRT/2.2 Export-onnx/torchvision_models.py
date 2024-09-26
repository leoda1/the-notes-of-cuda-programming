import torch
import torchvision
import onnxsim
import onnx
import os

def get_model(type):
    models = {
        "resnet": torchvision.models.resnet18(),
        "vgg": torchvision.models.vgg11(),
        "mobilenet": torchvision.models.mobilenet_v3_small(),
        "efficientnet": torchvision.models.efficientnet_b0(),
        "efficientnetv2": torchvision.models.efficientnet_v2_s(),
        "regnet": torchvision.models.regnet_x_1_6gf(),
        "shufflenetV2": torchvision.models.shufflenet_v2_x0_5(),
    }
    file = f"{type}.onnx"
    return models[type], file

def export_norm_onnx(model, file, input):
    model.cuda()
    torch.onnx.export(
        model=model, 
        args=(input,),
        f=file,
        input_names=["input0"],
        output_names=["output0"],
        opset_version=15)
    print(f"Finished normal onnx export for {file}")

    model_onnx = onnx.load(file)
    onnx.checker.check_model(model_onnx)
    print(f"Simplifying with onnx-simplifier {onnxsim.__version__} for {file}...")
    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "Simplification check failed"
    onnx.save(model_onnx, file)

def main(dir):
    input = torch.rand(1, 3, 224, 224, device='cuda')
    model_types = ["resnet", "vgg", "mobilenet", "efficientnet", "efficientnetv2", "regnet", "shufflenetV2"]
    for model_type in model_types:
        model, filename = get_model(model_type)
        full_path = os.path.join(dir, filename)
        export_norm_onnx(model, full_path, input)

if __name__ == "__main__":
    dir = "torchvision_models"  # 指定目录，无需通过命令行参数
    os.makedirs(dir, exist_ok=True)  # 创建目录如果不存在
    main(dir)
