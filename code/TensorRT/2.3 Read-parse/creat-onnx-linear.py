from onnx import helper
import onnx 

# -ModelProto 整体模型信息
# ---GraphProto 整个网络信息图
# -----NodeProto 各个节点
# -----AttributeProto 节点属性
# -----ValueInfoProto 节点输入输出信息

# -TensorProto 张量信息
# ---DataType 数据类型
# ---ShapeProto 张量形状
# ---float_data 张量数据

def create_onnx_model():
    #创建ValueInfoProto
    a = helper.make_tensor_value_info('a', onnx.TensorProto.FLOAT, [10, 10])
    b = helper.make_tensor_value_info('b', onnx.TensorProto.FLOAT, [10, 10])
    x = helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [10, 10])
    y = helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [10, 10])
    #创建NodeProto
    mul = helper.make_node('Mul', ['a', 'x'], 'c', "multiply")
    add = helper.make_node('Add', ['c', 'b'], 'y', "add")
    #创建GraphProto
    graph = helper.make_graph([mul, add], "sample-linear", [a, x, b], [y])
    #创建ModelProto
    model = helper.make_model(graph)
    #检查model是否有错误
    onnx.checker.check_model(model)
    #save model
    onnx.save(model, "models/sample-linear.onnx")
    return model

if __name__ == '__main__':
    model = create_onnx_model()