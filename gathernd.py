import numpy as np
import onnx
from onnx import AttributeProto, GraphProto, OperatorSetIdProto, TensorProto, helper, numpy_helper  # noqa: F401
import onnxruntime as ort

MODEL_NAME = "gathernd"
data_ = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
                 ).reshape((2, 2, 2)).astype(np.int32)
indices_ = np.array([[1], [0]]).reshape((2, 1)).astype(np.int64)


def make_model(batch_dims):
    nodes = []
    data = helper.make_tensor_value_info("data", TensorProto.INT32, [2, 2, 2])
    indices = helper.make_tensor_value_info(
        "indices", TensorProto.INT64, [2, 1])
    output = 0
    if batch_dims == 0:
        output = helper.make_tensor_value_info(
            "output", TensorProto.INT32, [2, 2, 2])
    if batch_dims == 1:
        output = helper.make_tensor_value_info(
            "output", TensorProto.INT32, [2, 2])

    gathernd_node = helper.make_node(
        "GatherND",
        ["data", "indices"],
        ["output"],
        name="gathernd_2",
        batch_dims=batch_dims,
    )
    nodes.append(gathernd_node)

    graph_def = helper.make_graph(
        nodes,
        "test-model",
        [data, indices],
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
    onnx.save(model_def, MODEL_NAME + str(batch_dims) + ".onnx")


print("From 0:")
print(data_[0:])
print(data_[0:].shape)
batch_dims = 0
make_model(batch_dims)
ort_sess = ort.InferenceSession(MODEL_NAME + str(batch_dims) + ".onnx")
outputs = ort_sess.run(None, {'data': data_, 'indices': indices_})
print(outputs)


print("From 1:")
print(data_[1:])
print(data_[1:].shape)

batch_dims = 1
make_model(batch_dims)
ort_sess = ort.InferenceSession(MODEL_NAME + str(batch_dims) + ".onnx")
outputs = ort_sess.run(None, {'data': data_, 'indices': indices_})
print(outputs)
