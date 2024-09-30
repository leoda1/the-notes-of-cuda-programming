import onnx_graphsurgeon as gs
import numpy as np
import onnx
import os
if not os.path.exists("models"):
    os.makedirs("models")
# def load_model(model : onnx.ModelProto):
#     graph = gs.import_onnx(model)
#     print(graph.inputs)
#     print(graph.outputs)

def main() -> None:
    model = onnx.load("models/swin-tiny.onnx")
    graph = gs.import_onnx(model)
    tensors = graph.tensors()

    #LayerNorm部分
    print(tensors["/patch_embed/Transpose_output_0"])
    print(tensors["/patch_embed/norm/LayerNormalization_output_0"])
    # 去onnx图中的相应模块地方 找到上一个output的name 然后将这个tensor复制进来
    graph.inputs = [
        tensors["/patch_embed/Transpose_output_0"].to_variable(dtype = np.float32, shape = (1, 3136, 128))
    ]
    # 同理输出
    graph.outputs = [
        tensors["/patch_embed/norm/LayerNormalization_output_0"].to_variable(dtype = np.float32, shape = (1, 3136, 128))
    ]
    graph.cleanup()
    onnx.save(gs.export_onnx(graph), "models/LN-swin-subgraph.onnx")

    #MultiHeadAttention部分
    graph = gs.import_onnx(model)
    tensors = graph.tensors()
    print(tensors["/layers.0/blocks.0/Reshape_3_output_0"])
    print(tensors["/layers.0/blocks.0/attn/proj/MatMul_output_0"])
    graph.inputs = [
        tensors["/layers.0/blocks.0/Reshape_3_output_0"].to_variable(dtype = np.float32, shape = (64, 49, 128))
    ]
    graph.outputs = [
        tensors["/layers.0/blocks.0/attn/proj/MatMul_output_0"].to_variable(dtype = np.float32, shape = (64, 49, 128))
    ]
    graph.cleanup()
    onnx.save(gs.export_onnx(graph), "models/MHSA-swin-subgraph.onnx")

if __name__ == "__main__":
    main()