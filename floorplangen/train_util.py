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
                 lr_anneal_steps=0):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.schedule_sampler = UniformSampler(diffusion)
        self.lr_anneal_steps = lr_anneal_steps
        self.step = 1 # 0から始めると最後の保存時に辻褄あわなくなるし，1からにする
        self.save_interval = save_interval
        self.log_interval = log_interval

        self.opt = AdamW(
            self.model.parameters(),
            lr=self.lr
        )


    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step < self.lr_anneal_steps + 1
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
                    
            
            # モデル保存
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
            
            
            self.step += 1
            
        # Save the last checkpoint if it wasn't already saved.
        if self.step % self.save_interval != 0:
            self.save()
    
    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.opt.step()

    def forward_backward(self, batch, cond):
        batch = batch.to(dist_util.dev())
        cond = {k: v.to(dist_util.dev()) for k, v in cond.items()}
        self.opt.zero_grad()
        # sample t from uniform distribution 1~1000. weights are all 1
        t, weights = self.schedule_sampler.sample(batch.shape[0], dist_util.dev())
        terms = self.diffusion.training_losses(self.model, batch, t, model_kwargs=cond)
        loss = terms["loss"].mean()
        print(f'loss: {loss}')
        loss.backward()
    
    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.model.state_dict()
            print('saving model {rate}....')

            filename = f"model{(self.step):06d}.pt"

            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)
        
        save_checkpoint(0, self.model.parameters())

