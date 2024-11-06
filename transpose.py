import onnxruntime as ort
import onnx
from onnx import helper as helper
from onnx import TensorProto as tp
import numpy as np



def buildAndRunBinaryGraph():
    op = 'Transpose'
    DATA_TYPE = tp.FLOAT
    comment = 'float'

    def getNPType(dataType):
        if (dataType == tp.UINT32):
            return np.uint32
        elif (dataType == tp.INT32):
            return np.int32
        else:
            return np.float32

    NP_TYPE = getNPType(DATA_TYPE)

    #t1 = np.array([1024, 1024]).astype(NP_TYPE)
    t1 =  np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape((1, 3, 4, 1)).astype(np.float32)
    # The functional nodes:
    n1 = helper.make_node(op, inputs=['a'], outputs=['output'], name='n1')
    n1.attribute.extend([helper.make_attribute("perm", [0,2,1,3])])
    # Create the graph
    g1 = helper.make_graph([n1], 'preprocessing',
                           [helper.make_tensor_value_info('a', DATA_TYPE, [1, 3, 4, 1])],
                           [helper.make_tensor_value_info('output', DATA_TYPE, [1, 4, 3, 1])])
    # Create the model and check
    print(100)
    m1 = helper.make_model(g1, producer_name='onnxtranspose-demo')
    m1.ir_version = 9
    #onnx.checker.check_model(m1)
    # Save the model
    MODEL_NAME = op+'_'+ comment +'.onnx'
    onnx.save(m1, MODEL_NAME)
    ort_sess = ort.InferenceSession(MODEL_NAME)
    outputs = ort_sess.run(None, {'a': t1})
    # [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12],
    print(outputs)
    return MODEL_NAME

print(buildAndRunBinaryGraph())
