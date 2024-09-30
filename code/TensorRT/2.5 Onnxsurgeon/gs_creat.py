import onnx
import onnx_graphsurgeon as gs
import numpy as np
import os
if not os.path.exists("models"):
    os.makedirs("models")
#======================================================================================
#                                   gs简单创建Conv
#======================================================================================
# gs中的IR有三种结构
# Tensor：
#     两种类型：
#         Variable：不到推理阶段不知道的变量
#         Constant: 训练阶段就已知的变量
# Node：
#         与helper中的NodeProto对应，表示节点
# Graph：
#         与helper中的GraphProto对应，表示图
def main() -> None:
    input = gs.Variable(
            name  = "input",
            dtype = np.float32,
            shape = (1, 3, 224, 224))
    weight = gs.Constant(
            name = "weight",
            values = np.random.randn(5, 3, 3, 3))
    bias   = gs.Constant(
            name = "bias",
            values = np.random.randn(5))
    output = gs.Variable(
            name  = "output",
            dtype = np.float32,
            shape = (1, 5, 224, 224))
    node   = gs.Node(
            op      = "Conv",
            name    = "conv",
            attrs   = {"pads":[1, 1, 1, 1]},
            inputs  = [input, weight, bias],
            outputs = [output])
    graph = gs.Graph(
            nodes   = [node],
            inputs  = [input],
            outputs = [output],)
    model = gs.export_onnx(graph)
    
    onnx.save(model, "models/conv.onnx")
#======================================================================================
#                              graph中注册调用的函数                    
#======================================================================================
def register_custom_op():
    @gs.Graph.register()
    def add(self, a, b):
        return self.layer(op="Add", inputs=[a, b], outputs=["add_out_gs"])
    
    @gs.Graph.register()
    def mul(self, a, b):
        return self.layer(op="Mul", inputs=[a, b], outputs=["mul_out_gs"])
    
    @gs.Graph.register()
    def gemm(self, a, b, trans_a = False, trans_b = False):
        attrs = {"transA": int(trans_a), "transB": int(trans_b)}
        return self.layer(op="Gemm", inputs=[a, b], outputs=["gemm_out_gs"], attrs=attrs)
    
    @gs.Graph.register()
    def relu(self, x):
        return self.layer(op="Relu", inputs=[x], outputs=["relu_out_gs"])

    @gs.Graph.register()
    def sigmoid(self, x):
        return self.layer(op="Sigmoid", inputs=[x], outputs=["sigmoid_out_gs"])
    

    @gs.Graph.register()
    def softmax(self, x):
        return self.layer(op="Softmax", inputs=[x], outputs=["softmax_out_gs"])

    @gs.Graph.register()
    def concat(self, inputs, axis):
        return self.layer(op="Concat", inputs=inputs, outputs=["concat_out_gs"], attrs={"axis": axis})

    @gs.Graph.register()
    def reshape(self, x, shape):
        return self.layer(op="Reshape", inputs=[x], outputs=["reshape_out_gs"], attrs={"shape": shape})

    graph = gs.Graph(opset = 12)
    MA = gs.Constant(name="MA", values=np.random.randn(64, 32))
    MB = gs.Constant(name="MB", values=np.random.randn(64, 32))
    MC = gs.Constant(name="MC", values=np.random.randn(64, 32))
    MD = gs.Constant(name="MD", values=np.random.randn(64, 32))
    input = gs.Variable(name="input", dtype=np.float32, shape=(64, 64))

    gemm0 = graph.gemm(input, MA, trans_b=True)
    relu0 = graph.relu(*graph.add(*gemm0, MB))
    mul0 = graph.mul(*relu0, MC)
    output = graph.add(*mul0, MD)

    graph.inputs = [input]
    graph.outputs = output

    for out in graph.outputs:
        out.dtype = np.float32

    onnx.save(gs.export_onnx(graph), "models/register-graph.onnx")

#======================================================================================
#                                    创建子图                    
#======================================================================================


if __name__ == "__main__":
    # ========================
    # graphsurgeon简单创建conv
    # ========================
    # main()

    # ========================
    # graph中注册调用的函数
    # ========================
    register_custom_op()

    # ========================
    # 创建子图
    # ========================
    #subgraph()
