from mltk.utils.archive_downloader import download_verify_extract
import os
import sys
from mltk.utils.path import create_tempdir
import onnxsim
import onnx

###########ONNX Model
cifar10_matlab_model_example_dir = './fuzz_report_1/bug-Symptom.EXCEPTION-Stage.COMPILATION-2/model.onnx'

###########Configure Paths

# This contains the path to the pre-trained model in ONNX model format
# For this tutorial, we use the one downloaded from above
# Update this path to point to your specific model if necessary


#ONNX_MODEL_PATH = './fuzz_report_1/bug-Symptom.EXCEPTION-Stage.COMPILATION-2/model.onnx'

ONNX_MODEL_PATH = '../nnsmith_output/model.onnx'


# This contains the path to our working directory where all
# generated, intermediate files will be stored.
# For this tutorial, we use a temp directory.
# Update as necessary for your setup

#WORKING_DIR = create_tempdir('model_onnx_to_tflite')
WORKING_DIR = create_tempdir('model_onnx_to_tflite')

# Use the filename for the model's name
MODEL_NAME = os.path.basename(ONNX_MODEL_PATH)[:-len('.onnx')]

###########Simplify the ONNX model

simplified_onnx_model, success = onnxsim.simplify(ONNX_MODEL_PATH)
assert success, 'Failed to simplify the ONNX model. You may have to skip this step'
simplified_onnx_model_path =  f'{WORKING_DIR}/{MODEL_NAME}.simplified.onnx'

print(f'Generating {simplified_onnx_model_path} ...')
onnx.save(simplified_onnx_model, simplified_onnx_model_path)
print('done')

###########Convert to OpenVino Intermediate Format



# Import the model optimizer tool from the openvino_dev package
from openvino.tools.mo import main as mo_main
import onnx
from onnx_tf.backend import prepare
from mltk.utils.shell_cmd import run_shell_cmd


# Load the ONNX model
#onnx_model = onnx.load(ONNX_MODEL_PATH)
onnx_model = onnx.load(simplified_onnx_model_path)
tf_rep = prepare(onnx_model)


# Get the input tensor shape
input_tensor = tf_rep.signatures[tf_rep.inputs[0]]
input_shape = input_tensor.shape
input_shape_str = '[' + ','.join([str(x) for x in input_shape]) + ']'

#openvino_out_dir = f'{WORKING_DIR}/openvino'
openvino_out_dir='/data/nexpnnsmith/onnx2ov'
#os.makedirs(openvino_out_dir, exist_ok=True)


cmd = [ 
    sys.executable, mo_main.__file__, 
    '--input_model', simplified_onnx_model_path,
    '--input_shape', input_shape_str,
    '--output_dir', openvino_out_dir,
    #'--data_type', 'FP32'

]
retcode, retmsg = run_shell_cmd(cmd,  outfile=sys.stdout)
assert retcode == 0, 'Failed to do conversion' 

###########Convert from OpenVino to TF-Lite-Float32

from mltk.utils.shell_cmd import run_shell_cmd

openvino2tensorflow_out_dir = '/data/nexpnnsmith/openvino2tensorflow'
openvino_xml_name = os.path.basename(simplified_onnx_model_path)[:-len('.onnx')] + '.xml'


if os.name == 'nt':
  openvino2tensorflow_exe_cmd = [sys.executable, os.path.join(os.path.dirname(sys.executable), 'openvino2tensorflow')]
else:
  openvino2tensorflow_exe_cmd = ['openvino2tensorflow']

print(f'Generating openvino2tensorflow model at: {openvino2tensorflow_out_dir} ...')
cmd = openvino2tensorflow_exe_cmd + [ 
    '--model_path', f'{openvino_out_dir}/{openvino_xml_name}',
    '--model_output_path', openvino2tensorflow_out_dir,
    '--output_saved_model',
    '--output_no_quant_float32_tflite'
]

retcode, retmsg = run_shell_cmd(cmd)
assert retcode == 0, retmsg
print('done')