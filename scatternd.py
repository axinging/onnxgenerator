import numpy as np
import onnx
from onnx import AttributeProto, GraphProto, OperatorSetIdProto, TensorProto, helper, numpy_helper  # noqa: F401
import onnxruntime as ort

import onnxruntime as ort
import onnx
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
    return np.float32

MODEL_NAME = "scatternd"

TPTYPE = TensorProto.FLOAT
NPTYPE = getNPType(TPTYPE)

data_ = np.array([1.1, 2.2, 3.1, 4.5, 5.3, 6.1, 7.8, 8.9]
                 ).reshape((8)).astype(NPTYPE)
indices_ = np.array([4, 3, 1, 7]).reshape((1, 4, 1)).astype(np.int64)
updates_ = np.array([9.1, 10.2, 11.3, 12.5]).reshape((1, 4)).astype(NPTYPE)

def make_model():
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
        name="scatternd_demo"
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
    onnxdomain.version = 13
    # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
    onnxdomain.domain = ""
    opsets.append(onnxdomain)

    kwargs = {}
    kwargs["opset_imports"] = opsets

    model_def = helper.make_model(
        graph_def, producer_name="onnx-example", **kwargs)
    onnx.save(model_def, MODEL_NAME + ".onnx")


make_model()
ort_sess = ort.InferenceSession(MODEL_NAME + ".onnx")
outputs = ort_sess.run(None, {'data': data_, 'indices': indices_, 'updates': updates_})
print(outputs)
