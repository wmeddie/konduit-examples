import argparse
import kfserving
import os
from typing import Dict
import torch
import importlib
import sys

PYTORCH_FILE = "model.pt"
DEFAULT_MODEL_NAME = "model"
DEFAULT_LOCAL_MODEL_DIR = "/tmp/model"

class PyTorchModel(kfserving.KFModel):
    def __init__(self, name: str, model_class_name: str, model_dir: str):
        super().__init__(name)
        self.name = name
        self.model_class_name = model_class_name
        self.model_dir = model_dir
        self.ready = False
        self._pytorch = None

    def load(self):
        model_file_dir = kfserving.Storage.download(self.model_dir)
        model_file = os.path.join(model_file_dir, PYTORCH_FILE)
        py_files = []
        for filename in os.listdir(model_file_dir):
            if filename.endswith('.py'):
                py_files.append(filename)
        if len(py_files) == 1:
            model_class_file = os.path.join(model_file_dir, py_files[0])
        elif len(py_files) == 0:
            raise Exception('Missing PyTorch Model Class File.')
        else:
            raise Exception('More than one Python file is detected',
                            'Only one Python file is allowed within model_dir.')
        model_class_name = self.model_class_name

        # Load the python class into memory
        sys.path.append(os.path.dirname(model_class_file))
        modulename = os.path.basename(model_class_file).split('.')[0].replace('-', '_')
        model_class = getattr(importlib.import_module(modulename), model_class_name)

        # Make sure the model weight is transform with the right device in this machine
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._pytorch = model_class().to(device)
        self._pytorch.load_state_dict(torch.load(model_file, map_location=device))
        self._pytorch.eval()
        self.ready = True

    def predict(self, request: Dict) -> Dict:
        inputs = []
        try:
            inputs = torch.tensor(request["instances"])
        except Exception as e:
            raise Exception(
                "Failed to initialize Torch Tensor from inputs: %s, %s" % (e, inputs))
        try:
            return {"predictions":  self._pytorch(inputs).tolist()}
        except Exception as e:
            raise Exception("Failed to predict %s" % e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
    parser.add_argument('--model_dir', required=True, help='A URI pointer to the model directory')
    args, _ = parser.parse_known_args()

    model = PyTorchModel("cifar10", "Model", args.model_dir)
    model.load()
    kfserving.KFServer().start([model])