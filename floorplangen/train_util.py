import copy
import time
import torch as th
from torch.optim import AdamW
import blobfile as bf
import dist_util
import logger

from resample import UniformSampler

def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
        return logger.get_dir()

class TrainLoop:
    def __init__(self,
                 model,
                 diffusion,
                 data, 
                 batch_size,
                 lr,
                 save_interval=100,
                 log_interval=100,
                 use_fp16=False,
                 weight_decay=0.0,
                 lr_anneal_steps=0):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.schedule_sampler = UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.step = 1 # 0から始めると最後の保存時に辻褄あわなくなるし，1からにする
        self.resume_step = 0
        self.save_interval = save_interval
        self.log_interval = log_interval
        # self.global_batch = self.batch_size * dist.get_world_size()

        self.opt = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.start_time = time.time() # 開始時間の記録
        # if self.resume_step:
        #     self._load_optimizer_state()
        #     # Model was resumed, either due to a restart or a checkpoint
        #     # being specified at the command line.
        #     self.ema_params = [
        #         self._load_ema_parameters(rate) for rate in self.ema_rate
        #     ]
        # else:
        #     self.ema_params = [
        #         copy.deepcopy(self.mp_trainer.master_params)
        #         for _ in range(len(self.ema_rate))
        #     ]
    def _load_optimizer_state(self):
        # main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        # find_resume_checkpoint() just return None
        main_checkpoint = self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)



    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps + 1
        ):
            # batch.shep = [batch_size, 2, 100]
            # cond is a dict that has keys ['door_mask',
            #                    'self_mask',
            #                    'gen_mask',
            #                    'room_types',
            #                    'corner_indices',
            #                    'room_indices',
            #                    'src_key_padding_mask',
            #                    'connections',
            #                    'graph']
            batch, cond = next(self.data)
            self.run_step(batch, cond)

            # 100000step毎にlearning rateを更新(0.1倍)
            if self.step % 100000 == 0:
                lr = self.lr * (0.1**(self.step//100000))
                logger.log(f"Step {self.step}: Updating learning rate to {lr}")
                for param_group in self.opt.param_groups:
                    param_group["lr"] = lr
            # ログ
            if self.step % self.log_interval == 0:
                elapsed_time = (time.time() - self.start_time) / 60  # 経過時間を分単位で計算
                logger.log(f"Step {self.step}: {elapsed_time:.2f} minutes elapsed since start.")
                logger.dumpkvs()
            # モデル保存
            if self.step % self.save_interval == 0:
                self.save()
            
            self.step += 1
            
        # Save the last checkpoint if it wasn't already saved.
        if self.step % self.save_interval != 0:
            self.save()
    
    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.opt.step()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        batch = batch.to(dist_util.dev())
        cond = {k: v.to(dist_util.dev()) for k, v in cond.items()}
        self.opt.zero_grad()
        # sample t from uniform distribution 1~1000. weights are all 1
        t, weights = self.schedule_sampler.sample(batch.shape[0], dist_util.dev())
        losses = self.diffusion.training_losses(self.model, batch, t, model_kwargs=cond)
        loss = losses["loss"].mean()
        # print(f'loss: {loss}')
        log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
        loss.backward()
    
    def _anneal_lr(self):
        # 線形的に学習率を下げる
        if not self.lr_anneal_steps: # None or 0の時return
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps # 現在のstepが全体における割合(進捗度)
        lr = self.lr * (1 - frac_done)  
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
    
    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        elapsed_time = (time.time() - self.start_time) / 60  # 経過時間を分単位で計算
        logger.logkv("elapsed_time_minutes", elapsed_time)
        # logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    
    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.model.state_dict()
            print(f'saving model {rate}....')

            filename = f"model{(self.step+self.resume_step):06d}.pt"

            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)
        
        save_checkpoint(0, self.model.parameters())

def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)