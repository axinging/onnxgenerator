import onnxruntime as ort
import onnx
from onnx import helper as helper
from onnx import TensorProto as tp
import numpy as np


import onnx
from onnx import helper, shape_inference
from onnx import TensorProto
#import onnx.optimizer
import onnxruntime as rt


import math
import numpy as np
import scipy.stats as stats


def getOptLevelString(sess_options):
    optLevel = sess_options.graph_optimization_level
    if sess_options.graph_optimization_level == rt.GraphOptimizationLevel.ORT_ENABLE_BASIC:
        return 'basic'
    elif sess_options.graph_optimization_level == rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED: 
        return 'extended'
    elif sess_options.graph_optimization_level == rt.GraphOptimizationLevel.ORT_ENABLE_ALL: 
        return 'all'
    else:
        return 'none'

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


def buildAndRunBinaryGraph(MODEL_NAME, VERSION):
 
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

    W1 =  helper.make_tensor(name="W1", data_type=onnx.TensorProto.FLOAT, dims=W1_dims, vals=truncated_normal(W1_dims))

    B1_dims = [8]

    B1 =  helper.make_tensor(name="B1", data_type=onnx.TensorProto.FLOAT, dims=B1_dims, vals=zeros(B1_dims))


    node1 = helper.make_node('Conv', inputs=['X', 'W1', 'B1'], outputs=['T1'], kernel_shape=[5,5], strides=[1,1], pads=[2,2,2,2])
    node2 = helper.make_node('Relu', inputs=['T1'], outputs=['T2'])
#
    #clip_attr = helper.make_attribute("max", 200.0)
    #node2.attribute.append(clip_attr)
    #clip_attr = helper.make_attribute("min", 0.0)
    #node2.attribute.append(clip_attr)

    graph = helper.make_graph(
        [node1, node2],
        'mnist_conv',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, ([batch_size, 1, 28, 28])),
        helper.make_tensor_value_info('W1', TensorProto.FLOAT, W1_dims),
        helper.make_tensor_value_info('B1', TensorProto.FLOAT, B1_dims),
        ],
        [helper.make_tensor_value_info('T2', TensorProto.FLOAT, ([batch_size, 8, 28, 28]))],
        [W1, B1]
    )
    #original_model = helper.make_model(graph, producer_name='onnx-examples')
    original_model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", VERSION)])
    onnx.checker.check_model(original_model)
    
    MODEL_PATH = "./models/"
    onnx.save(original_model, MODEL_PATH+MODEL_NAME+".onnx")
    sess_options = rt.SessionOptions()
    # Set graph optimization level
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

    # To enable model serialization after graph optimization set this
    levelString = getOptLevelString(sess_options)
    sess_options.optimized_model_filepath = MODEL_PATH + MODEL_NAME + "-opt-py-"+levelString+".onnx"
    session = rt.InferenceSession(MODEL_PATH+MODEL_NAME+".onnx", sess_options)
    #outputs = session.run(None, {'a': t1, 'b': t2})
    # Print Result
    #print(type(DATA_TYPE).__name__, outputs[0])

VERSION = 1
MODEL_NAME = 'conv_simple'
# buildAndRunBinaryGraph(MODEL_NAME+str(VERSION), VERSION)

VERSION = 9
MODEL_NAME = 'conv_simple'
buildAndRunBinaryGraph(MODEL_NAME+str(VERSION), VERSION)

VERSION = 11
MODEL_NAME = 'conv_simple'
buildAndRunBinaryGraph(MODEL_NAME+str(VERSION), VERSION)

VERSION = 12
MODEL_NAME = 'conv_simple'
buildAndRunBinaryGraph(MODEL_NAME+str(VERSION), VERSION)

VERSION = 21
MODEL_NAME = 'conv_simple'
buildAndRunBinaryGraph(MODEL_NAME+str(VERSION), VERSION)
