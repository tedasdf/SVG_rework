# # ## Basic Setup
import os , importlib, logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PosixPath
from typing import Optional

import modal
from pydantic import BaseModel


MINUTES = 60  # seconds
HOURS = 60 * MINUTES

### Hyperparameters 
APP_NAME = "example-encoder-training-run"
VOLUME_NAME = APP_NAME + '-volume'
GPU = 'A10G'
LOG_DIR = "logs"


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
    .uv_pip_install(requirements=["../requirements_modal.txt"])
)

torch_image = torch_image.add_local_file("../requirements_modal.txt", remote_path="/root/requirements.txt")
torch_image = torch_image.add_local_dir(
    Path(__file__).parent , remote_path="/root/src"
)


with torch_image.imports():
    
    import glob
    # import os
    # from timeit import default_timer as timer

    from pytorch_lightning import seed_everything
    # import tensorboard
    # import torch
    # from src.dataset import Dataset
    # from src.logs_manager import LogsManager
    # from src.model import AttentionModel
    # from src.tokenizer import Tokenizer


# We also have some local dependencies that we'll need to import into the remote environment.
# We add them into the remote container.


# We can also "pre-import" libraries that will be used by the functions we run on Modal in a given image
# using the `with image.imports` context manager.


@app.function(image=torch_image)
def check_deps():
    return check_requirements('/root/requirements.txt')



# @app.function(
#     image=torch_image,
#     volumes={volume_path: volume},
#     gpu=gpu,
#     timeout=1 * HOURS,
# )
# def train_model(
#     node_rank,
#     n_nodes,
#     hparams,
#     experiment_name,
#     run_to_first_save=False,
#     n_steps=3000,
#     n_steps_before_eval=None,
#     n_steps_before_checkpoint=None,
# ):
#     # optimizer, data, and model prep
#     batch_size = 64
#     learning_rate = 3e-4

#     n_eval_steps = 100
#     if n_steps_before_eval is None:
#         n_steps_before_eval = int(n_steps / 8)  # eval eight times per run
#     if n_steps_before_checkpoint is None:
#         n_steps_before_checkpoint = int(n_steps / 4)  # save four times per run

#     train_percent = 0.9

#     L.basicConfig(
#         level=L.INFO,
#         format=f"\033[0;32m%(asctime)s %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] [Node {node_rank + 1}] %(message)s\033[0m",
#         datefmt="%b %d %H:%M:%S",
#     )

#     # use GPU if available
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     L.info("Remote Device: %s // GPU: %s", device, gpu)

#     input_file_path = volume_path / "shakespeare_char.txt"
#     text = prepare_data(input_file_path, volume)

#     # construct tokenizer & dataset
#     tokenizer = Tokenizer(text)
#     dataset = Dataset(
#         tokenizer.encode(text),
#         train_percent,
#         batch_size,
#         hparams.context_size,
#         device,
#     )

#     # build the model
#     model = build_model(hparams, tokenizer.vocab_size, device)
#     num_parameters = sum(p.numel() for p in model.parameters())
#     L.info(f"Num parameters: {num_parameters}")

#     optimizer = setup_optimizer(model, learning_rate)

#     # TensorBoard logging & checkpointing prep
#     logs_manager = LogsManager(experiment_name, hparams, num_parameters, tb_log_path)
#     L.info(f"Model name: {logs_manager.model_name}")

#     model_save_dir = model_save_path / experiment_name / logs_manager.model_name
#     if model_save_dir.exists():
#         L.info("Loading model from checkpoint...")
#         checkpoint = torch.load(str(model_save_dir / model_filename))
#         is_best_model = not run_to_first_save
#         if is_best_model:
#             make_best_symbolic_link(model_save_dir, model_filename, experiment_name)
#         model.load_state_dict(checkpoint["model"])
#         start_step = checkpoint["steps"] + 1
#     else:
#         model_save_dir.mkdir(parents=True, exist_ok=True)
#         start_step = 0
#         checkpoint = init_checkpoint(model, tokenizer, optimizer, start_step, hparams)

#     checkpoint_path = model_save_dir / model_filename

#     out = training_loop(
#         start_step,
#         n_steps,
#         n_steps_before_eval,
#         n_steps_before_checkpoint,
#         n_eval_steps,
#         dataset,
#         tokenizer,
#         model,
#         optimizer,
#         logs_manager,
#         checkpoint,
#         checkpoint_path,
#         run_to_first_save,
#     )

#     return node_rank, float(out["val"]), hparams


@app.local_entrypoint()
def main():
    # check dependencies
    result_future = check_deps.remote()

    # HYPERPARAMETER

    default_args = {
        "name": "",                  # -n / --name
        "resume": "",                # -r / --resume
        "base": [],                  # -b / --base
        "train": False,              # -t / --train
        "no_test": False,            # --no-test
        "project": None,             # -p / --project
        "debug": False,              # -d / --debug
        "seed": 23,                  # -s / --seed
        "scale_lr": True,            # --scale_lr
        "config_path": "svg/configs/example_svg_autoencoder_vitsp.yaml" # <----- Config file 
    }

    # name 
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    cfg_fname = os.path.split(default_args)[-1]
    cfg_name = os.path.splitext(cfg_fname)[0]
    name = cfg_name + now

    ##### MIGHT CHANGE TO modal specific ###########
    logdir = os.path.join(LOG_DIR, cfg_name)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    ################################################
    
    logging.basicConfig(
        level=logging.INFO,
        format=f"\033[0;32m%(asctime)s %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] [Node {node_rank + 1}] %(message)s\033[0m",
        datefmt="%b %d %H:%M:%S",
    )  

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Remote Device: %s // GPU: %s", device, gpu)

    # create multiple ckpt_files
    ckpt_files = glob.glob(os.path.join(logdir, "checkpoints", "epoch=*.ckpt"))
    if not ckpt_files:
        print(f"Warning: No checkpoint files found in {os.path.join(logdir, 'checkpoints')}, training from scratch")
        ckpt = None
    else:
        ckpt_files.sort(key=lambda x: int(os.path.basename(x).split('=')[1].split('.')[0]))
        ckpt = ckpt_files[-1]
    default_args.resume_from_checkpoint = ckpt
    
    # read from configs
    configs = OmegaConf.load(default_args.base)

    # train config with all lighting varibales
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config = dict(
        accelerator=trainer_config.accelerator, 
        devices=trainer_config.devices, 
        num_nodes=trainer_config.num_nodes, 
        strategy=trainer_config.strategy, 
        max_epochs=trainer_config.max_epochs, 
        precision=trainer_config.precision if hasattr(trainer_config, 'precision') else 32
    )
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # set seed 
    seed_everything(default_args.seed)
    # model
    model = instantiate_from_config(config.model)

    # if config.init_weight is not None, load the weights
    try:
        print(f"Loading initial weights from {config.init_weight}")
        model.load_state_dict(torch.load(config.init_weight)['state_dict'], strict=False)
    except:
        print(f"There is no initial weights to load.")

    ############ callbacks ############
    trainer_kwargs = dict()

    default_logger_cfg = {
        "target": "pytorch_lightning.loggers.TensorBoardLogger",
        "params": {
            "name": "tensorboard",
            "save_dir": logdir,
        }
    }

    logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
        }
    }
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = -1
    
    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg =  OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
    if version.parse(pl.__version__) < version.parse('1.4.0'):
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "main_svg_autoencoder.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            }
        },
        "image_logger": {
            "target": "main_svg_autoencoder.ImageLogger",
            "params": {
                "batch_frequency": 750,
                "max_images": 4,
                "clamp": True
            }
        },
        "learning_rate_logger": {
            "target": "main_svg_autoencoder.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                # "log_momentum": True
            }
        },
        "cuda_callback": {
            "target": "main_svg_autoencoder.CUDACallback"
        },
    }


     if version.parse(pl.__version__) >= version.parse('1.4.0'):
        default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
        print(
            'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint':
                {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                'params': {
                    "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                    "filename": "{epoch:06}-{step:09}",
                    "verbose": True,
                    'save_top_k': -1,
                    'every_n_train_steps': 10000,
                    'save_weights_only': True
                }
                }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
        callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
    elif 'ignore_keys_callback' in callbacks_cfg:
        del callbacks_cfg['ignore_keys_callback']

    print(callbacks_cfg)
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    trainer = Trainer(**{k: v for k, v in vars(trainer_opt).items() if v is not None}, **trainer_kwargs)
    trainer.logdir = logdir  ###
    

if __name__ == '__main__':
    # print(Path(__file__).parent)
    
    # if os.path.exists('../requirements_new.txt'):
    #     print("YIPPIY")
    

    # result = check_requirements()
    # print(result)
    import os 
    cfg_fname = os.path.split("svg/configs/example_svg_autoencoder_vitsp.yaml")[-1]
    cfg_name = os.path.splitext(cfg_fname)[0]
    print(cfg_name)