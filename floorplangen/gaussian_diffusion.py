import numpy as np
import torch as th
from tqdm import tqdm

class GaussianDiffusion:
    def __init__(self):
        self.num_timesteps = 1000
        betas = get_named_beta_schedule('linear', 1000)
        self.betas = betas
        alphas = 1.0 - betas
        # alphaの累積積
        # 拡散過程に要するαや，その平方根，1-αの平方根の各ステップの値を格納している．
        # これらの値を使って，q(x_t | x_0)を計算する
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
    
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # 式84
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        # 式84のそれぞれの係数 coef1 -> x0, coef2->xt
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        
    
    def training_losses(self, model, x_start, t, model_kwargs):
        # ノイズ生成
        noise = th.randn_like(x_start)
        # tステップ分のノイズ負荷
        x_t = self.q_sample(x_start, t, noise)
        terms = {}
        
        # 損失関数MSEを計算
        # x_tと，ノイズに対するスケーリング係数を取得
        xtalpha = _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape).permute([0, 2, 1])
        epsalpha = _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape).permute([0,2,1])
        model_output_dec, model_output_bin = model(x_t, t, xtalpha=xtalpha, epsalpha=epsalpha, **model_kwargs)
        
        # 分散の学習をする場合を想定→とおもったけどどうもエラーが出てしまうので飛ばす
        # Diffusionは分散の学習は不要説あるよね？あとで勉強する
        # B, C = x_t.shape[:2] # batch_size, channels
        # model_output, model_var_values = th.split(model_output_dec, C, dim=1)
        # frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
        # return frozen_out, x_start, x_t, t
        
        # 損失を計算するためのtarget(正解)
        # 元コードだと，x_t-1，x_start, noiseをtargetにできるけどデフォルトはnoiseなので，今回はnoiseをtargetとする
        # return x_start, noise, t, model_output_dec, model_output_bin
        
        # 離散化=====================
        def dec2bin(xinp, bits):
            mask = 2 ** th.arange(bits - 1, -1, -1).to(xinp.device, xinp.dtype)
            return xinp.unsqueeze(-1).bitwise_and(mask).ne(0).float()
    
        target = noise # targetは本来xt-1, x_start, noiseと選べるが，デフォルトではnoiseなので，ここではnoiseのみ
        bin_target = x_start.detach()
        bin_target = (bin_target/2 + 0.5) # -> [0,1]
        bin_target = bin_target * 256 #-> [0, 256]
        bin_target = dec2bin(bin_target.permute([0,2,1]).round().int(), 8) 
        bin_target = bin_target.reshape([target.shape[0], target.shape[2], 16]).permute([0,2,1])
        t_weights = (t<10).unsqueeze(1).unsqueeze(2)
        t_weights = t_weights * (t_weights.shape[0]/max(1, t_weights.sum()))
        bin_target[bin_target==0] = -1
        # ===========================

        # 損失計算=====================
        # これら二つの損失を足す
        terms = {}
        tmp_mask = (1 - model_kwargs['src_key_padding_mask'])
        # t_weightsは，t<10の場合のみ有効．それ以外は0になるのでmse_binは考慮されない
        terms["mse_bin"] = mean_flat(((bin_target - model_output_bin) ** 2) * t_weights, tmp_mask)
        terms["mse_dec"] = mean_flat(((target - model_output_dec) ** 2), tmp_mask)
        terms["loss"] = terms["mse_dec"] + terms["mse_bin"]

        return terms

    
    def q_sample(self, x_start, t, noise):
        # sample from q(x_t | x_0).
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        # 式84
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_sample_loop(self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """

        myfinal = []

        for i, sample in tqdm(enumerate(self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress
        ))):
            if i>969: # 最後の30stepを保存
                myfinal.append(sample['sample'])
        return th.stack(myfinal)

    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        # 初期ノイズXT
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            indices = tqdm(indices)
        
        # t=T->1
        for i in indices:
            # バッチ分
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def p_sample(
            self,
            model,
            x,
            t,
            clip_denoised=None,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # z. q(xt-1|xt, x0)の式(84)でサンプリングする際に，xt-1 ~ μ+σ*zという形でサンプリングする
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) -1)))
        ) # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart":out["pred_xstart"]}

    def p_mean_variance(
            self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        B, C = x.shape[:2]
        assert t.shape == (B,)
        xtalpha = _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape).permute([0,2,1])
        epsalpha = _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape).permute([0,2,1])

        # [TODO]本来，p_sampleが呼ばれる時はevalなので，dataloaderもeval用になり，condにpredix=syn_がつくが，evalのDatasetを実装ちゃんとできてないので，is_syn=False
        model_output_dec, model_output_bin = model(x, t, xtalpha=xtalpha, epsalpha=epsalpha, is_syn=False, **model_kwargs)
        model_output = model_output_dec

        predict_descrete = 32


        # 以下はまだよくわかってない
        if t[0] < predict_descrete:
            def bin2dec(b, bits):
                mask = 2 ** th.arange(bits - 1, -1, -1).to(b.device, b.dtype)
                return th.sum(mask * b, -1)
            model_output_bin[model_output_bin>0] = 1
            model_output_bin[model_output_bin<=0] = 0
            model_output_bin = bin2dec(model_output_bin.round().int().permute([0,2,1]).reshape(model_output_bin.shape[0],
                model_output_bin.shape[2], 2, 8), 8).permute([0,2,1])

            model_output_bin = ((model_output_bin/256) - 0.5) * 2
            model_output = model_output_bin
        
        # 分散の算出
        # t=0の時はαbarが1になるので，分散が0になってしまう．それを避けるために，最初の要素は
        # posterior_variance[1]にし，それ以降は，βが非常に小さい場合((1-α_{t-1})/(1-1-α_t))は１に近似する形でβのみとしてるs
        model_variance = np.append(self.posterior_variance[1], self.betas[1:])
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))

        # 特定のtの値だけを取得(&tensorにする)
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
            
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if t[0] >= predict_descrete:
            # ノイズ予測のモデルならこれ．式115でxtとεからx0を計算する
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
            # 平均は真の分布q(xt-1|xt, x0)の式展開から求められる(式84)
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            pred_xstart = process_xstart(model_output)
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart
        }
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def mean_flat(tensor, padding_mask):
    """
    Take the mean over all non-batch dimensions.
    """
    tensor = tensor * padding_mask.unsqueeze(1)
    tensor = tensor.mean(dim=list(range(1, len(tensor.shape))))/th.sum(padding_mask, dim=1)
    return tensor

# cosineも本来あるが，ここではsimpleにlinearのみ実装
def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == 'linear':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )

