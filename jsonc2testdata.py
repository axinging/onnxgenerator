import sys
import ort_test_dir_utils
import onnx_test_data_utils
import numpy as np
import onnx
import os
import onnxruntime as ort
import onnx
from onnx import helper as helper
from onnx import TensorProto as tp
import numpy as np

'''
data  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                           1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                           1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                           1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
'''
data  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
size2 = 3
def buildAndRunBinaryGraph():
    op = 'Transpose'
    DATA_TYPE = tp.FLOAT
    comment = 'float'

    #t1 =  np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape((1, 3, 4, 1)).astype(np.float32)
    t1 =  np.array(data).reshape((1, size2, 4, 1)).astype(np.float32)
    # The functional nodes:
    n1 = helper.make_node(op, inputs=['data'], outputs=['output'], name='n1')
    n1.attribute.extend([helper.make_attribute("perm", [0,2,1,3])])
    # Create the graph
    g1 = helper.make_graph([n1], 'preprocessing',
                           [helper.make_tensor_value_info('data', DATA_TYPE, [1, size2, 4, 1])],
                           [helper.make_tensor_value_info('output', DATA_TYPE, [1, 4, size2, 1])])
    # Create the model and check
    m1 = helper.make_model(graph = g1, producer_name='onnxtranspose-demo', opset_imports=[helper.make_operatorsetid("ai.onnx", 9)])
    m1.ir_version = 3
    #onnx.checker.check_model(m1)
    # Save the model
    MODEL_NAME = op+'_'+ comment +'.onnx'
    onnx.save(m1, MODEL_NAME)
    ort_sess = ort.InferenceSession(MODEL_NAME)
    outputs = ort_sess.run(None, {'data': t1})
    # [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12],
    print(outputs)
    return MODEL_NAME


inputs = {}
#inputs['data'] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape((1, 3, 4, 1)).astype(np.float32)
inputs['data'] = np.array(data).reshape((1, size2, 4, 1)).astype(np.float32)

model_path = buildAndRunBinaryGraph()
print(model_path)
patht = "test_transpose_broadcasetfail"
ort_test_dir_utils.create_test_dir(model_path, 'temp/'+ patht, '', name_input_map=inputs)

# can easily dump the input and output to visually check it's as expected
onnx_test_data_utils.dump_pb('temp/'+patht+'/test_data_set_0')
os.rename('temp/'+patht + '/'+model_path,'temp/'+patht +'/model.onnx')
'''
Shape: (1, 4, 3, 1)
[[[[ 1.]
   [ 5.]
   [ 9.]]

  [[ 2.]
   [ 6.]
   [10.]]

  [[ 3.]
   [ 7.]
   [11.]]

  [[ 4.]
   [ 8.]
   [12.]]]]
'''
