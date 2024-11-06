import onnxruntime as ort
import onnx
from onnx import helper as helper
from onnx import TensorProto as tp
import numpy as np

def getNPType(dataType):
    if (dataType == tp.UINT32):
        return np.uint32
    elif (dataType == tp.UINT16):
        return np.uint16
    elif (dataType == tp.INT32):
        return np.int32
    elif (dataType == tp.FLOAT16):
        return np.float16
    else:
        return np.float32

def buildAndRunBinaryGraph(op, DATA_TYPE, comment):
    NP_TYPE = getNPType(DATA_TYPE)

    t1 = np.array([4, 8, 9]).astype(NP_TYPE)
    t2 = np.array([1, 3, 9]).astype(NP_TYPE)
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
    # Create the model and check
    m1 = helper.make_model(g1, producer_name='onnxsub-demo')
    onnx.checker.check_model(m1)
    # Save the model
    MODEL_NAME = op+'_' + type(DATA_TYPE).__name__ +'_'+ comment +'.onnx'
    onnx.save(m1, MODEL_NAME)
    ort_sess = ort.InferenceSession(MODEL_NAME)
    outputs = ort_sess.run(None, {'a': t1, 'b': t2})
    # Print Result
    print(type(DATA_TYPE).__name__, outputs[0])

op = 'Sub'
DATA_TYPE = tp.FLOAT16
buildAndRunBinaryGraph(op, DATA_TYPE, 'FLOAT16')


