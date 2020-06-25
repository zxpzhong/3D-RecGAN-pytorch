import os
import torch
import torch.onnx as torch_onnx
from torch.autograd import Variable


def pytorch2onnx(input_shape = None,input_model=None, output_model_name = None):


    if output_model_name == None or input_model==None or input_shape == None:
        print('Transfer model pytorch2onnx model_name is None')
        exit(-6)


    # Use this an input trace to serialize the model
    model_onnx_path = output_model_name

    # Export the model to an ONNX file
    dummy_input = Variable(torch.randn(input_shape))
    output = torch_onnx.export(input_model,
                               dummy_input,
                               model_onnx_path,
                               verbose=True)
    print("Export of torch_model.onnx complete!")


import onnx
import caffe2.python.onnx.backend

# Prepare the inputs, here we use numpy to generate some random inputs for demo purpose
import numpy as np


def onnx2caffe(input_shape, onnx_path, caffe_path):
    img = np.random.randn(input_shape).astype(np.float32)

    # Load the ONNX model
    model = onnx.load(onnx_path)
    # Run the ONNX model with Caffe2
    outputs = caffe2.python.onnx.backend.run_model(model, [img])


if __name__ == '__main__':
    pass
