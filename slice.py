import onnxruntime as ort
import onnx
from onnx import helper as helper
from onnx import TensorProto as tp
import numpy as np

from json import JSONEncoder
import numpy

import json
from json import JSONEncoder
import numpy

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def numpyTypeAsString(dataType):
    if (dataType == np.int32):
        return "int32"
    elif (dataType == np.float32):
        return ""
    else:
        return ""

def createTensorsJson(inputs, outputs):
    inputArrayJson = []
    nameStr = ""
    for input in inputs:  # [::-1]:
        inputData = input if isinstance(input, numpy.ndarray) else str(input)
        inputArrayJson.insert(len(inputArrayJson), {
                              "data": inputData, "dims": input.shape, "type": str(input.dtype)})
        nameStr += "T["+str(input.shape).replace('(',
                                                 '').replace(')', '').rstrip(',') + "] "
    nameStr += "(" + str(inputs[0].dtype)+")"
    #outputJson = ({"data": output, "dims": output.shape,
    #              "type": str(output.dtype)})

    outputArrayJson = []    
    for output in outputs:  # [::-1]:
        outputData = output if isinstance(output, numpy.ndarray) else str(output)
        outputArrayJson.insert(len(outputArrayJson), {
                              "data": outputData, "dims": output.shape, "type": str(output.dtype)})
        nameStr += "T["+str(output.shape).replace('(',
                                                 '').replace(')', '').rstrip(',') + "] "
    return {"name": nameStr, "inputs": inputArrayJson, "outputs": outputArrayJson}


def createJsonFromTensors(testName, opName, caseInfos, suffix):
    cases = []
    for caseInfo in caseInfos:
        cases.insert(len(cases), createTensorsJson(
            caseInfo["inputs"], caseInfo["outputs"]))
    testJson = {"name": testName,  "operator": opName, "attributes": [], "cases": cases}
    # use dump() to write array into file
    encodedNumpyData = json.dumps([testJson], cls=NumpyArrayEncoder)

    # Writing to sample.json
    if (suffix != ""):
        suffix = "_" + suffix
    with open(opName.lower() + suffix + ".jsonc", "w") as outfile:
        outfile.write(encodedNumpyData)



def getNPType(dataType):
    if (dataType == tp.UINT32):
        return np.uint32
    elif (dataType == tp.INT32):
        return np.int32
    else:
        return np.float32

def export_slice_default_axes(DATA_TYPE) -> None:
    NP_TYPE = getNPType(DATA_TYPE)
    print('str(NP_TYPE)        ::::::::::', str(NP_TYPE))
    node = onnx.helper.make_node(
        "Slice",
        inputs=["x", "starts", "ends"],
        outputs=["y"],
    )
    XD = 1
    YD = 1
    x = np.random.randn(5).astype(NP_TYPE)
    print('x.shape: ', x.shape)
    starts = np.array([3], dtype=np.int64)
    print('starts.shape: ',starts.shape)
    ends = np.array([4], dtype=np.int64)
    print('ends.shape: ', ends.shape)
    y = x[3:4]
    print('y.shape: ', y.shape, y)

    # Create the graph
    g1 = helper.make_graph([node], 'preprocessing',
                           [helper.make_tensor_value_info(
                               'x', DATA_TYPE, [5]), helper.make_tensor_value_info('starts', tp.INT64, [1]), helper.make_tensor_value_info('ends', tp.INT64, [1])],
                           [helper.make_tensor_value_info('y', DATA_TYPE, [1])])

    m1 = helper.make_model(g1, producer_name='onnxsub-demo')
    onnx.checker.check_model(m1)
    # Save the model
    op = 'export_slice_default_axes'
    MODEL_NAME = op + '.onnx'
    onnx.save(m1, MODEL_NAME)
    ort_sess = ort.InferenceSession(MODEL_NAME)
    outputs = ort_sess.run(None, {'x': x, 'starts': starts, 'ends': ends})
    # Print Result
    print(type(DATA_TYPE).__name__, outputs[0])
    opName = "Slice"
    caseInfo1 = {"inputs": [x, starts, ends], "outputs": outputs}
    createJsonFromTensors(opName + " with no attributes", opName, [caseInfo1], numpyTypeAsString(NP_TYPE))

export_slice_default_axes(tp.FLOAT)
export_slice_default_axes(tp.INT32)

# numpyArray = numpy.array([[1, 22, 33, 77, 88, 99], [1, 22, 33, 77, 88, 99]])
# numpyArray2 = numpy.array([2, 77, 88, 99])
# numpyArrayOut = numpy.array([3, 77, 88, 99])
# numpyFloat = numpy.float32("0.1")
# 
# opName = "Add"
# caseInfo1 = {"inputs": [numpyArray, numpyArray2], "outputs": numpyArrayOut}
# caseInfo2 = {"inputs": [numpyArray, numpyFloat], "outputs": numpyArrayOut}
# createJsonFromTensors("Add with no attributes", "Add", [caseInfo1, caseInfo2])
