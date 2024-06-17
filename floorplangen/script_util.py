from gaussian_diffusion import GaussianDiffusion
from transformer import TransformerModel

def create_diffusion_and_transformer():
    '''
    input_channels = 2+8*2 # AU()により，隣接した点との間に7点=計9点の座標を取っている
    condition_channels = 89 # 25+32+32=89?
    model_channels = 128
    out_channels = 2 # 入力channelの2倍ぽい（平均と分散？->4にするとエラーになるのでとりあず2）
    dataset = None # 参照されてないみたい
    use_checkpoint = None # 参照されてないみたい
    use_unet = False # self.unet使われてなさそう
    analog_bit = False
    '''
    return GaussianDiffusion(), TransformerModel(in_channels=18,
                                                condition_channels=89,
                                                model_channels=128,
                                                out_channels=2,
                                                dataset=None,
                                                use_checkpoint=None,
                                                use_unet=False,
                                                analog_bit=False)
