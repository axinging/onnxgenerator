import onnxruntime as ort
import onnx
from onnx import helper as helper
from onnx import TensorProto as tp
import numpy as np


def getNPType(dataType):
    if (dataType == tp.UINT32):
        return np.uint32
    elif (dataType == tp.INT32):
        return np.int32
    else:
        return np.float32

def buildAndRunBinaryGraph(op, DATA_TYPE, comment):
    NP_TYPE = getNPType(DATA_TYPE)

    #t1 = np.array([1024, 1024]).astype(NP_TYPE)
    t1 = np.random.randn(16, 16).astype(NP_TYPE)
    # The functional nodes:
    n1 = helper.make_node(op, inputs=['A', 'B'], outputs=['Y'], name='n1')
    # Create the graph
    g1 = helper.make_graph([n1], 'preprocessing',
                           [helper.make_tensor_value_info('A', DATA_TYPE, [16, 16]), helper.make_tensor_value_info('B', DATA_TYPE, [16, 16])],
                           [helper.make_tensor_value_info('Y', DATA_TYPE, [16, 16])])
    # Create the model and check
    print(100)
    m1 = helper.make_model(g1, producer_name='onnxtranspose-demo')
    #m1.ir_version = 9
    #onnx.checker.check_model(m1)
    # Save the model
    MODEL_NAME = op+'_' + type(DATA_TYPE).__name__ +'_'+ comment +'.onnx'
    onnx.save(m1, MODEL_NAME)
    print(200)
    ort_sess = ort.InferenceSession(MODEL_NAME)
    outputs = ort_sess.run(None, {'a': t1})
    print(900)

    # Print Result
    print(type(DATA_TYPE).__name__, outputs[0])
    return m1

op = 'Matmul'
print('FLOAT', tp.FLOAT)

#DATA_TYPE = tp.FLOAT
#buildAndRunBinaryGraph(op, DATA_TYPE, 'FLOAT')

DATA_TYPE = tp.FLOAT
buildAndRunBinaryGraph(op, DATA_TYPE, 'FLOAT')

DATA_TYPE = tp.UINT32
#buildAndRunBinaryGraph(op, DATA_TYPE, 'UINT32')
