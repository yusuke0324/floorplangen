import sys
import os
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
logger.configure()
logger.log("creating model and diffusion...")
diffusion, transformer = create_diffusion_and_transformer()
logger.log("creating data loader...")
batch_size = 16
analog_bit = False
target_set = 8
set_name = 'train'
data = load_rplanhg_data(
            batch_size=batch_size,
            analog_bit=analog_bit,
            target_set=target_set,
            set_name=set_name,
        )
logger.log("training...")
transformer.to(dist_util.dev())
TrainLoop(transformer,
          diffusion,
          data,
          batch_size=8,
          lr=0.001,
          lr_anneal_steps=100000,
          save_interval=10000,
          log_interval=10000
         ).run_loop()