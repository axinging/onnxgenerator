import sys
import ort_test_dir_utils
import onnx_test_data_utils
import numpy as np
import onnx
import os

def getInput(model_path):
    #model = onnx.load(r"jets-text-to-speech.onnx")
    model = onnx.load(model_path)

    # The model is represented as a protobuf structure and it can be accessed
    # using the standard python-for-protobuf methods

    # iterate through inputs of the graph
    inputInfo = []
    for input in model.graph.input:
        # get type of input tensor
        tensor_type = input.type.tensor_type
        inputInfo.append({"name": input.name, "type": input.type})
        # check if it has a shape:
        if (tensor_type.HasField("shape")):
            return tensor_type.shape.dim
        else:
            print ("unknown rank", end="")
    return inputInfo

# example model with two float32 inputs called 'input1' (dims: {2, 1}) and 'input2' (dims: {'dynamic', 4})
model_path = './models/op_test_generated_model_Where_with_no_attributes.onnx'
inputInfo = getInput(model_path)
print(inputInfo[0]['name'])
inputs = {}
'''
auto condition_values = {true, false, true};  // std::initializer_list<bool> for OpTester::AddInput<bool>()
test.AddInput<bool>("condition", {1, 1, 3}, condition_values);
test.AddInput<T>("X", {1, 3, 1}, X_values);
test.AddInput<T>("Y", {3, 1, 1}, Y_values);
'''

inputs[inputInfo[0]['name']] = np.array([True, False, True]).reshape((1, 1, 3)).astype(np.bool)
inputs[inputInfo[1]['name']] = np.array([1.0, 1.0, 1.0]).reshape((1, 3, 1)).astype(np.float32)
inputs[inputInfo[2]['name']] = np.array([0.0,0.0, 0.0]).reshape((3, 1, 1)).astype(np.float32)
  #print(value)
# when using the default data generation any symbolic dimension values must be provided
symbolic_vals = {'dynamic':2} # provide value for symbolic dim named 'dynamic' in 'input2'

# let create_test_dir create random input in the (arbitrary) default range of -10 to 10.
# it will create data of the correct type based on the model.
#ort_test_dir_utils.create_test_dir(model_path, 'temp/test_where_broadcaset', 'test1', symbolic_dim_values_map=symbolic_vals)

# alternatively some or all input can be provided directly. any missing inputs will have random data generated.
# symbolic dimension values are only required for input data that is randomly generated,
# so we don't need to provide that in this case as we're explicitly providing all inputs.
patht = "test_where_broadcasetfail"
ort_test_dir_utils.create_test_dir(model_path, 'temp/test_where_broadcasetfail', '', name_input_map=inputs)

# can easily dump the input and output to visually check it's as expected
onnx_test_data_utils.dump_pb('temp/test_where_broadcasetfail/test_data_set_0')
os.rename('temp/test_where_broadcasetfail/op_test_generated_model_Where_with_no_attributes.onnx','temp/test_where_broadcasetfail/model.onnx')
