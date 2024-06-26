import sys
import os
import json

# 現在のスクリプトのディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# /work/scripts ディレクトリを追加
sys.path.append(os.path.join(current_dir))

from rplan_datautil import load_rplanhg_data
from rplan_datautil import RPlanhgDataset
from train_util import TrainLoop
from script_util import create_diffusion_and_transformer
from gaussian_diffusion import GaussianDiffusion
import logger
import dist_util

# ハイパーパラメータ設定
hyperparams = {
    "batch_size": 256, # 原論文では512
    "analog_bit": False,
    "target_set": 8, # 論文では5, 6, 7, 8の４グループ
    "set_name": 'train',
    "learning_rate": 0.001, # 論文は1e-3. 100kstep毎に*0.1
    "lr_anneal_steps": 250000, # 論文は250k
    "save_interval": 25000,
    "log_interval": 25000,
    "in_channels": 18,
    "condition_channels": 89,
    "model_channels": 128,
    "out_channels": 2,
    "dataset": None,
    "use_checkpoint": None,
    "use_unet": False,
    "analog_bit": False,
    "use_boundary": True
}

logger.configure()
logger.log("Hyperparameters: " + json.dumps(hyperparams, indent=4)) # ハイパーパラメータをログに記録
logger.log("creating model and diffusion...")

diffusion, transformer = create_diffusion_and_transformer(in_channels=hyperparams["in_channels"],
                                    condition_channels=hyperparams["condition_channels"],
                                    model_channels=hyperparams["model_channels"],
                                    out_channels=hyperparams["out_channels"],
                                    dataset=hyperparams["dataset"],
                                    use_checkpoint=hyperparams["use_checkpoint"],
                                    use_unet=hyperparams["use_unet"],
                                    analog_bit=hyperparams["analog_bit"],
                                    use_boundary=hyperparams["use_boundary"])
logger.log("creating data loader...")

data = load_rplanhg_data(
            batch_size=hyperparams["batch_size"],
            analog_bit=hyperparams["analog_bit"],
            target_set=hyperparams["target_set"],
            set_name=hyperparams["set_name"],
        )

logger.log("training...")
transformer.to(dist_util.dev())

TrainLoop(transformer,
          diffusion,
          data,
          batch_size=hyperparams["batch_size"],
          lr=hyperparams["learning_rate"],
          lr_anneal_steps=hyperparams["lr_anneal_steps"],
          save_interval=hyperparams["save_interval"],
          log_interval=hyperparams["log_interval"]
         ).run_loop()
