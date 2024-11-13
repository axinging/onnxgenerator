import numpy as np
import onnx
from onnx import AttributeProto, GraphProto, OperatorSetIdProto, TensorProto, helper, numpy_helper  # noqa: F401
import onnxruntime as ort

from onnx import helper as helper
from onnx import TensorProto as tp
import numpy as np

def getNPType(tpType):
    if (tpType == TensorProto.UINT32):
        return np.uint32
    elif (tpType == TensorProto.INT32):
        return np.int32
    elif (tpType == TensorProto.FLOAT):
        return np.float32
    elif (tpType == TensorProto.FLOAT16):
        return np.float16
    return np.float32

MODEL_NAME = "scatternd-reduction"


def make_model(reduction, TPTYPE):
    NPTYPE = getNPType(TPTYPE)
    nodes = []
    data = helper.make_tensor_value_info("data", TPTYPE, [8])
    indices = helper.make_tensor_value_info(
        "indices", TensorProto.INT64, [1, 4, 1])
    updates = helper.make_tensor_value_info(
        "updates", TPTYPE, [1, 4])
    output = 0
    output = helper.make_tensor_value_info(
            "output", TPTYPE, [8])

    gathernd_node = helper.make_node(
        "ScatterND",
        ["data", "indices", "updates"],
        ["output"],
        reduction=reduction,
        name="scatternd_demo",
    )
    nodes.append(gathernd_node)

    graph_def = helper.make_graph(
        nodes,
        "test-model",
        [data, indices, updates],
        [output],
    )

    opsets = []
    onnxdomain = OperatorSetIdProto()
    onnxdomain.version = 16
    # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
    onnxdomain.domain = ""
    opsets.append(onnxdomain)

    kwargs = {}
    kwargs["opset_imports"] = opsets

    model_def = helper.make_model(
        graph_def, producer_name="onnx-example", **kwargs)
    onnx.save(model_def, MODEL_NAME + ".onnx")


def buildAndRunBinaryGraph(reduction, TPTYPE = TensorProto.FLOAT):
    
    NPTYPE = getNPType(TPTYPE)
    make_model(reduction, TPTYPE)

    d1 =[1.1, 2.2, 3.1, 4.5, 5.3, 6.1, 7.8, 8.9] if  TPTYPE == TensorProto.FLOAT or TPTYPE == TensorProto.FLOAT16  else [1, 2, 3, 4, 5, 6, 7, 8]
    d2 =[9.1, 10.2, 11.3, 12.5] if  TPTYPE == TensorProto.FLOAT else  [9, 10, 11, 12]

    data_ = np.array(d1).reshape((8)).astype(NPTYPE)
    indices_ = np.array([4, 3, 1, 7]).reshape((1, 4, 1)).astype(np.int64)
    updates_ = np.array(d2).reshape((1, 4)).astype(NPTYPE)

    ort_sess = ort.InferenceSession(MODEL_NAME + ".onnx")
    outputs = ort_sess.run(None, {'data': data_, 'indices': indices_, 'updates': updates_})
    print(outputs)
    return MODEL_NAME + ".onnx"

#buildAndRunBinaryGraph('add', TensorProto.FLOAT)
buildAndRunBinaryGraph('mul', TensorProto.FLOAT16)
#buildAndRunBinaryGraph('add', TensorProto.INT32)
#buildAndRunBinaryGraph('mul', TensorProto.INT32)
