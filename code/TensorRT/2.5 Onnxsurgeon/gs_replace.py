#####################通过注册的函数进行创建网络########################
#          input (5, 5)             input (5, 5)
#            |                           |
#         identity                    identity 
#            |                           |
#           min           变成          clip
#            |                           |
#           max                     identity  
#            |                           | 
#         identity                  output (5, 5)
#            |
#          output (5, 5)
######################################################################
import onnx
import torch
import onnx_graphsurgeon as gs
import numpy as np
import onnxruntime
import os
if not os.path.exists("models"):
    os.makedirs("models")

@gs.Graph.register()
def min(self, *args):
    return self.layer(op = "Min", inputs = args, outputs = ["min_output"])

@gs.Graph.register()
def max(self, *args):
    return self.layer(op = "Max", inputs = args, outputs = ["max_output"])

@gs.Graph.register()
def identity(self, a):
    return self.layer(op="Identity", inputs=[a], outputs=["identity_output"])

@gs.Graph.register()
def clip(self, inputs, outputs):
    return self.layer(op="Clip", inputs=inputs, outputs=outputs) 

def create_onnx_graph():
    graph    = gs.Graph(opset=12)
    # 初始化网络需要用的参数
    min_val  = np.array(0, dtype=np.float32)
    max_val  = np.array(1, dtype=np.float32)
    input0   = gs.Variable(name="input0", dtype=np.float32, shape=(5, 5))
    # 设计网络架构
    identity0 = graph.identity(input0)
    min0      = graph.min(*identity0, max_val)
    max0      = graph.max(*min0, min_val)
    output0   = graph.identity(*max0)
    # 设置网络的输入输出
    graph.inputs = [input0]
    graph.outputs = output0
    # 设置网络的输出的数据类型
    for out in graph.outputs:
        out.dtype = np.float32
    # 保存模型
    onnx.save(gs.export_onnx(graph), "models/minmax.onnx")

def change_onnx_graph():
    graph = gs.import_onnx(onnx.load_model('models/minmax.onnx'))
    tensors = graph.tensors()

    inputs = [tensors["identity_output_0"], 
              tensors["onnx_graphsurgeon_constant_5"],
              tensors["onnx_graphsurgeon_constant_2"]]

    outputs = [tensors["max_output_6"]]
    
    # 因为要替换子网，所以需要把子网和周围的所有节点都断开联系
    for item in inputs:
        # print(item.outputs)
        item.outputs.clear()

    for item in outputs:
        # print(item.inputs)
        item.inputs.clear()

    # 通过注册的clip，重新把断开的联系链接起来
    graph.clip(inputs, outputs)

    # 删除所有额外的节点
    graph.cleanup()

    onnx.save(gs.export_onnx(graph), "models/minmax-clip.onnx")

#####################验证模型##########################################
def validate_onnx_graph(input, path):
    sess   = onnxruntime.InferenceSession(path)
    output = sess.run(None, {'input0': input.numpy()})

    print("input is \n", input)
    print("output is \n", output)

    
def main() -> None:
    input  = torch.Tensor(5, 5).uniform_(-1, 1)

    # 创建一个minmax的网络
    create_onnx_graph()

    # 通过onnxruntime确认导出onnx是否正确生成
    print("\nBefore modification:")
    validate_onnx_graph(input, "models/minmax.onnx")

    # 将minmax网络修改成clip网络
    change_onnx_graph()

    # 确认网络修改的结构是否正确
    print("\nAfter modification:")
    validate_onnx_graph(input, "models/minmax-clip.onnx")


if __name__ == "__main__":
    main()