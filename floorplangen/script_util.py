from gaussian_diffusion import GaussianDiffusion
from transformer import TransformerModel

def create_diffusion_and_transformer(in_channels=18,
                                    condition_channels=89,
                                    model_channels=128,
                                    out_channels=2,
                                    dataset=None,
                                    use_checkpoint=None,
                                    use_unet=False,
                                    analog_bit=False,
                                    use_boundary=True
):
    '''
    input_channels = 2+8*2 # AU()により，隣接した点との間に7点=計9点の座標を取っている
    condition_channels = 91 # 25+32+32+2=91, (use_boundary=Falseの場合は89) 
    model_channels = 128
    out_channels = 2 # 入力channelの2倍ぽい（平均と分散？->4にするとエラーになるのでとりあず2）
    dataset = None # 参照されてないみたい
    use_checkpoint = None # 参照されてないみたい
    use_unet = False # self.unet使われてなさそう
    analog_bit = False
    use_boundary=True
    '''
    return GaussianDiffusion(), TransformerModel(in_channels=in_channels,
                                                condition_channels=condition_channels,
                                                model_channels=model_channels,
                                                out_channels=out_channels,
                                                dataset=dataset,
                                                use_checkpoint=use_checkpoint,
                                                use_unet=use_unet,
                                                analog_bit=analog_bit,
                                                use_boundary=use_boundary)
