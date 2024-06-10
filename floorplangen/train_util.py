import torch as th
from torch.optim import AdamW
from resample import UniformSampler
class TrainLoop:
    def __init__(self,
                 model,
                 diffusion,
                 data, 
                 batch_size,
                 lr,
                 use_fp16=False,
                 lr_anneal_steps=0):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.schedule_sampler = UniformSampler(diffusion)
        self.lr_anneal_steps = lr_anneal_steps
        self.step = 0

        self.opt = AdamW(
            self.model.parameters(),
            lr=self.lr
        )

        if th.cuda.is_available():
            self.device =  th.device(f"cuda")
        self.device = th.device("cpu")

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step < self.lr_anneal_steps
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
    
    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.opt.step()

    def forward_backward(self, batch, cond):
        self.opt.zero_grad()
        # sample t from uniform distribution 1~1000. weights are all 1
        t, weights = self.schedule_sampler.sample(batch.shape[0], self.device)
        self.diffusion.training_losses(self.model, batch, t, model_kwargs=cond)
