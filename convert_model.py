import os
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.framework import graph_io
import keras
from keras.models import load_model


# Clear any previous session.
tf.keras.backend.clear_session()

#FP32, FP16
CHOOSE_MODEL = ["MobileNetV2","VGG19","Xception"]
BASE_MODEL = CHOOSE_MODEL[2]
PRECISION = 'FP16'
MIN_SEGMENT_SIZE = 3
MAX_WORKSPACE = 25
base_path = f'./{BASE_MODEL}/'
model_fname = base_path + f'{BASE_MODEL}.h5'

save_pb_dir = base_path + f'{BASE_MODEL}_tensorrt_{PRECISION}_MIN{MIN_SEGMENT_SIZE}_MAX{MAX_WORKSPACE}/'




def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

# This line must be executed before loading Keras model.
keras.backend.set_learning_phase(0) 

model = load_model(model_fname)
session = keras.backend.get_session()

input_names = [t.op.name for t in model.inputs]
output_names = [t.op.name for t in model.outputs]

# Prints input and output nodes names, take notes of them.
print("######################")
print("Input/output names")
print(input_names, output_names)
print("######################")

# Create a frozen_graph model from the keras model
frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)

# Convert the frozen_graph model to tensorrt format
trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << MAX_WORKSPACE,
    precision_mode=PRECISION,
    minimum_segment_size=MIN_SEGMENT_SIZE,
    
)

graph_io.write_graph(trt_graph, save_pb_dir,
                     "trt_graph.pb", as_text=False)
                     
os.remove(save_pb_dir+'frozen_model.pb')
