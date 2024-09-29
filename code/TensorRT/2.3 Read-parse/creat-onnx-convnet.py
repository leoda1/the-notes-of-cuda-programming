import onnx
import numpy as np

def main():
    batch_size = 1; input_channels = 3; height = 224; width = 224; output_channels = 16
    input_shape  = [batch_size, input_channels, height, width]
    output_shape = [batch_size, output_channels, 1, 1]
    # 创建input & output
    model_input_name  = "input"
    model_output_name = "output"
    input  = onnx.helper.make_tensor_value_info(model_input_name,  onnx.TensorProto.FLOAT,  input_shape)
    output = onnx.helper.make_tensor_value_info(model_output_name, onnx.TensorProto.FLOAT, output_shape)
    # 创建convolution节点
    conv_output_name = "conv2d_output"
    conv_kernel = 3
    conv_pads = 1
    conv_weight     = np.random.rand(output_channels, input_channels, conv_kernel, conv_kernel)
    conv_bias       = np.random.rand(output_channels)
    conv_weight_name = "conv2d_weight"
    conv_weight_initializer = onnx.helper.make_tensor(
        name = conv_weight_name,
        data_type = onnx.TensorProto.FLOAT,
        dims = conv_weight.shape,
        vals = conv_weight.flatten().tolist(),
    )
    conv_bias_name = "conv2d_bias"
    conv_bias_initializer = onnx.helper.make_tensor(
        name = conv_bias_name,
        data_type = onnx.TensorProto.FLOAT,
        dims = conv_bias.shape,
        vals = conv_bias.flatten().tolist(),
    )
    conv_node = onnx.helper.make_node(
        op_type="Conv",
        inputs=[model_input_name, conv_weight_name, conv_bias_name],
        outputs=[conv_output_name],
        name = "conv2d",
        kernel_shape = [conv_kernel, conv_kernel],
        pads = [conv_pads, conv_pads, conv_pads, conv_pads],
    )
    # 创建batchnorm节点
    bn1_output_name = "bn1.output"
    
    bn1_scale = np.random.rand(output_channels); bn1_bias = np.random.rand(output_channels)
    bn1_mean = np.random.rand(output_channels);  bn1_var = np.random.rand(output_channels)

    bn1_scale_name = "bn1.scale";  bn1_bias_name = "bn1.bias"
    bn1_mean_name = "bn1.mean";    bn1_var_name = "bn1.var"

    bn1_scale_initializer = onnx.helper.make_tensor(
        name = bn1_scale_name,
        data_type = onnx.TensorProto.FLOAT,
        dims = bn1_scale.shape,
        vals = bn1_scale.flatten().tolist(),
    )
    bn1_bias_initializer = onnx.helper.make_tensor(
        name = bn1_bias_name,
        data_type = onnx.TensorProto.FLOAT,
        dims = bn1_bias.shape,
        vals = bn1_bias.flatten().tolist(),
    )
    bn1_mean_initializer = onnx.helper.make_tensor(
        name = bn1_mean_name,
        data_type = onnx.TensorProto.FLOAT,
        dims = bn1_mean.shape,
        vals = bn1_mean.flatten().tolist(),
    )
    bn1_var_initializer = onnx.helper.make_tensor(
        name = bn1_var_name,
        data_type = onnx.TensorProto.FLOAT,
        dims = bn1_var.shape,
        vals = bn1_var.flatten().tolist(),
    )

    bn1_node = onnx.helper.make_node(
        op_type = "BatchNormalization",
        inputs  = [conv_output_name, bn1_scale_name, bn1_bias_name, bn1_mean_name, bn1_var_name],
        outputs = [bn1_output_name],
        name = "BatchNormal",
    )

    # 创建Relu节点
    relu_output_name = "relu_output"

    relu1_node = onnx.helper.make_node(
        op_type="Relu",
        inputs=[bn1_output_name],
        outputs=[relu_output_name],
        name = "relu",
    )

    # 创建AVGPOOL节点

    avgpool_node = onnx.helper.make_node(
        op_type="GlobalAveragePool",
        inputs=[relu_output_name],
        outputs=[model_output_name],
        name = "avgpool",
    )

    # 创建graph
    graph = onnx.helper.make_graph(
        nodes = [conv_node, bn1_node, relu1_node, avgpool_node],
        name = "convnet",
        inputs = [input],
        outputs = [output],
        initializer= [
            conv_weight_initializer,
            conv_bias_initializer,
            bn1_scale_initializer,
            bn1_bias_initializer,
            bn1_mean_initializer,
            bn1_var_initializer
        ],
    )

    # 创建model
    model = onnx.helper.make_model(graph, producer_name="convnet-sample")
    model.opset_import[0].version = 12
    model = onnx.shape_inference.infer_shapes(model)# 推断shape
    onnx.checker.check_model(model) # 检查模型
    print("Successfully created {}.onnx".format(graph.name))
    onnx.save(model, "models/convnet-sample.onnx") # 保存模型

if __name__ == "__main__":
    main()