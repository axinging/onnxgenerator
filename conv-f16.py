import onnxruntime as ort
import onnx
from onnx import helper as helper
from onnx import TensorProto as tp
import numpy as np


import onnx
from onnx import helper, shape_inference
from onnx import TensorProto
#import onnx.optimizer


import math
import numpy as np
import scipy.stats as stats

def truncated_normal(dims):   
    size = 1
    for dim in dims:
        size *= dim

    mu, stddev = 0, 1/math.sqrt(size)
    lower, upper = -2 * stddev, 2 * stddev
    X = stats.truncnorm( (lower - mu) / stddev, (upper - mu) / stddev, loc = mu, scale = stddev)

    return X.rvs(size).tolist()

    
def zeros(dim):
    return [0] * dim[0]

def buildAndRunBinaryGraph(op, DATA_TYPE, comment):
 
    '''
    # The required constants:
    c1 = helper.make_node('Constant', inputs=[], outputs=['c1'], name='c1-node',
                          value=helper.make_tensor(name='c1v', data_type=DATA_TYPE,
                                                   dims=t1.shape, vals=t1.flatten()))
    # The functional nodes:
    n1 = helper.make_node(op, inputs=['a', 'b'], outputs=['output'], name='n1')
    # Create the graph
    g1 = helper.make_graph([n1], 'preprocessing',
                           [helper.make_tensor_value_info(
                               'a', DATA_TYPE, [3]), helper.make_tensor_value_info('b', DATA_TYPE, [3])],
                           [helper.make_tensor_value_info('output', DATA_TYPE, [3])])
    '''
    batch_size = 1

    W1_dims = [8, 1, 5, 5]
    W2_dims = [16, 8, 5, 5]
    W3_dims = [256, 10]

    W1 =  helper.make_tensor(name="W1", data_type=onnx.TensorProto.FLOAT, dims=W1_dims, vals=truncated_normal(W1_dims))
    W2 =  helper.make_tensor(name="W2", data_type=onnx.TensorProto.FLOAT, dims=W2_dims, vals=truncated_normal(W2_dims))
    W3 =  helper.make_tensor(name="W3", data_type=onnx.TensorProto.FLOAT, dims=W3_dims, vals=truncated_normal(W3_dims))

    B1_dims = [8]
    B2_dims = [16]
    B3_dims = [10]

    B1 =  helper.make_tensor(name="B1", data_type=onnx.TensorProto.FLOAT, dims=B1_dims, vals=zeros(B1_dims))
    B2 =  helper.make_tensor(name="B2", data_type=onnx.TensorProto.FLOAT, dims=B2_dims, vals=zeros(B2_dims))
    B3 =  helper.make_tensor(name="B3", data_type=onnx.TensorProto.FLOAT, dims=B3_dims, vals=zeros(B3_dims))


    shape = helper.make_tensor(name="shape", data_type=onnx.TensorProto.INT64, dims=[2], vals=[batch_size, 256])
    # Create the model and check

    node1 = helper.make_node('Conv', inputs=['X', 'W1', 'B1'], outputs=['T1'], kernel_shape=[5,5], strides=[1,1], pads=[2,2,2,2])
    node2 = helper.make_node('Relu', inputs=['T1'], outputs=['T2'])
    node3 = helper.make_node('MaxPool', inputs=['T2'], outputs=['T3'], kernel_shape=[2,2], strides=[2,2])

    node4 = helper.make_node('Conv', inputs=['T3', 'W2', 'B2'], outputs=['T4'], kernel_shape=[5,5], strides=[1,1], pads=[2,2,2,2])
    node5 = helper.make_node('Relu', inputs=['T4'], outputs=['T5'])
    node6 = helper.make_node('MaxPool', inputs=['T5'], outputs=['T6'], kernel_shape=[3,3], strides=[3,3])

    node7 = helper.make_node('Reshape', inputs=['T6', 'shape'], outputs=['T7'])

    node8 = helper.make_node('Gemm', inputs=['T7', 'W3', 'B3'], outputs=['predictions'])

    graph = helper.make_graph(
        [node1, node2, node3, node4, node5, node6, node7, node8],
        'mnist_conv',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, ([batch_size, 1, 28, 28])),
        helper.make_tensor_value_info('W1', TensorProto.FLOAT, W1_dims),
        helper.make_tensor_value_info('W2', TensorProto.FLOAT, W2_dims),
        helper.make_tensor_value_info('W3', TensorProto.FLOAT, W3_dims),
        helper.make_tensor_value_info('B1', TensorProto.FLOAT, B1_dims),
        helper.make_tensor_value_info('B2', TensorProto.FLOAT, B2_dims),
        helper.make_tensor_value_info('B3', TensorProto.FLOAT, B3_dims),
        helper.make_tensor_value_info('shape', TensorProto.INT64, [2]),
        ],
        [helper.make_tensor_value_info('predictions', TensorProto.FLOAT, ([batch_size, 10]))],
        [W1, W2, W3, B1, B2, B3, shape]
    )
    #original_model = helper.make_model(graph, producer_name='onnx-examples')
    original_model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 8)])

    MODEL_NAME = 'Conv_' + type(DATA_TYPE).__name__ +'_'+ comment +'.onnx'
    onnx.save(original_model, MODEL_NAME)
    ort_sess = ort.InferenceSession(MODEL_NAME, providers=["CPUExecutionProvider"])
    #outputs = ort_sess.run(None, {'a': t1, 'b': t2})
    # Print Result
    #print(type(DATA_TYPE).__name__, outputs[0])

op = 'Conv'
DATA_TYPE = tp.FLOAT16
buildAndRunBinaryGraph(op, DATA_TYPE, 'FLOAT16')


import onnxruntime as rt

sess_options = rt.SessionOptions()

# Set graph optimization level
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

def getOptLevelString(optLevel):
    if sess_options.graph_optimization_level == rt.GraphOptimizationLevel.ORT_ENABLE_BASIC:
        return 'basic'
    elif sess_options.graph_optimization_level == rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED: 
        return 'extended'
    elif sess_options.graph_optimization_level == rt.GraphOptimizationLevel.ORT_ENABLE_ALL: 
        return 'all'
    else:
        return 'none'
# To enable model serialization after graph optimization set this
levelString = getOptLevelString(sess_options.graph_optimization_level)
modelName = 'Conv_int_FLOAT16'
sess_options.optimized_model_filepath = modelName + "-opt-py-"+levelString+".onnx"

session = rt.InferenceSession(modelName + ".onnx", sess_options)


