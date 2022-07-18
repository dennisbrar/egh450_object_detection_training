import onnx
import argparse
import os

def arg_parse():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Adjust Yolov5.onnx output layers for depthai')

    parser.add_argument("--model", help = "Path to input onnx model", default = None, type = str)
    parser.add_argument("--output", help = "Name of altered output model", default = "YoloV5_depthai.onnx", type = str)

    return parser.parse_args()

def main():
    args = arg_parse()

    if not os.path.exists(args.model):
        raise FileNotFoundError("Model does not exist at path {}".format(args.model))
    
    output_path = os.path.join(os.getcwd(), args.output)

    if os.path.exists(output_path):
        raise RuntimeError("Output name {} already exists, choose a different name or remove exisiting file!".format(output_path))
    
    onnx_model = onnx.load(args.model)

    conv_indicies = []
    for i, n in enumerate(onnx_model.graph.node):
        if "Conv" in n.name:
            conv_indicies.append(i)

    input1, input2, input3 = conv_indicies[-3:]

    sigmoid1 = onnx.helper.make_node(
            'Sigmoid',
            inputs=[onnx_model.graph.node[input1].output[0]],
            outputs=['output1_yolov5'],
    )

    sigmoid2 = onnx.helper.make_node(
            'Sigmoid',
            inputs=[onnx_model.graph.node[input2].output[0]],
            outputs=['output2_yolov5'],
    )

    sigmoid3 = onnx.helper.make_node(
            'Sigmoid',
            inputs=[onnx_model.graph.node[input3].output[0]],
            outputs=['output3_yolov5'],
    )

    onnx_model.graph.node.append(sigmoid1)
    onnx_model.graph.node.append(sigmoid2)
    onnx_model.graph.node.append(sigmoid3)

    onnx.save(onnx_model, output_path)

if __name__ == '__main__':
    main()
