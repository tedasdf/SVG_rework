# # ## Basic Setup
import os , importlib
import logging as L
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PosixPath
from typing import Optional

import modal
from pydantic import BaseModel



#### HELPER FUNCTIONS
def check_requirements(req_file="../requirements_new.txt"):
    installed = {}

    # Explicitly specify encoding
    content = Path(req_file).read_text(encoding="utf-16")  # <-- change from default

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0].split(">")[0].split("<")[0]

        name_map = {
            "Pillow": "PIL",
            "opencv-python": "cv2",
        }
        import_name = name_map.get(pkg_name, pkg_name)

        try:
            importlib.import_module(import_name)
        except ImportError as e:
            installed[pkg_name] = f"ERROR: {e}"

    return installed
# #####


MINUTES = 60  # seconds
HOURS = 60 * MINUTES

### Hyperparameters 
APP_NAME = "example-encoder-training-run"
VOLUME_NAME = APP_NAME + '-volume'
GPU = 'A10G'


app = modal.App(APP_NAME)

# Since we'll be coordinating training across multiple machines we'll use a
# distributed [Volume](https://modal.com/docs/guide/volumes)
# to store the data, checkpointed models, and TensorBoard logs.

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
volume_path = Path("/vol/data")
model_filename = "nano_gpt_model.pt"
best_model_filename = "best_nano_gpt_model.pt"
tb_log_path = volume_path / "tb_logs"
model_save_path = volume_path / "models"

# ### Define dependencies in container images

# The container image for training  is based on Modal's default slim Debian Linux image with `torch`
# for defining and running our neural network and `tensorboard` for monitoring training.

torch_image  = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(requirements=["requirements_modal.txt"])
)

torch_image = torch_image.add_local_file("requirements_modal.txt", remote_path="/root/requirements.txt")

# We also have some local dependencies that we'll need to import into the remote environment.
# We add them into the remote container.


# We can also "pre-import" libraries that will be used by the functions we run on Modal in a given image
# using the `with image.imports` context manager.


@app.function(image=torch_image)
def check_deps():
    return check_requirements('/root/requirements.txt')

    # deps = ["torch", "tensorboard", "numpy", "pydantic"]
    # results = {}
    # for d in deps:
    #     try:
    #         __import__(d)
    #         results[d] = "OK"
    #     except ImportError as e:
    #         results[d] = str(e)

    # return results

@app.local_entrypoint()
def main():
    print("HEELO")
    result_future = check_deps.remote()
    result = result_future
    print(result)




# if __name__ == '__main__':
#     print(Path(__file__).parent)
    
#     if os.path.exists('../requirements_new.txt'):
#         print("YIPPIY")
    

#     result = check_requirements()
#     print(result)