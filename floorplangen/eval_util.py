import os
import sys
from PIL import Image
import numpy as np
import torch as th
from tqdm import tqdm
import drawSvg as drawsvg
import io
import cairosvg
# 現在のスクリプトのディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# /work/eval_util.py ディレクトリを追加
sys.path.append(os.path.join(current_dir))
from script_util import create_diffusion_and_transformer
from vis import vis_sample
import dist_util

def create_mask_from_image(image, target_color=(255, 255, 255)):
    # 画像から指定された色の領域を抽出してマスクを生成する
    # デフォルトはtarget_color=黒なので，vis_sampleのimages_colorに対応
    mask = Image.new('L', image.size, 0)
    data = np.array(image)
    
    # target_colorに一致するピクセルを白に、それ以外を黒にする
    mask_data = np.all(data[:, :, :3] == target_color, axis=-1).astype(np.uint8) * 255
    mask.putdata(mask_data.flatten())
    
    return mask

def plot_interpolated_boundary(model_kwargs, resolution=256, fill='none'):
    """
    model_kwargsのinterpolated_boundary_pointsを描画する関数
    
    Args:
    model_kwargs (dict): model_kwargs辞書
    resolution (int): 描画解像度
    """
    boundary_points = model_kwargs['interpolated_boundary_points']
    batch_images_boundary = []

    for i in tqdm(range(boundary_points.shape[0])):  # バッチ内の各サンプルについてループ
        draw = drawsvg.Drawing(resolution, resolution, displayInline=True)
        draw.append(drawsvg.Rectangle(0, 0, resolution, resolution, fill='white'))
        
        # ループの境界点を取得
        boundary_coords = boundary_points[i]
        # 座標を変換
        boundary_coords_transformed = (boundary_coords / 2 + 0.5) * resolution
        
        # 線を描画
        draw.append(drawsvg.Lines(
            *boundary_coords_transformed.flatten().tolist(),
            close=True, fill=fill, stroke='black', stroke_width=2))
        
        image_boundary = Image.open(io.BytesIO(cairosvg.svg2png(draw.asSvg())))
        batch_images_boundary.append(image_boundary)
    
    return batch_images_boundary

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou


class EvalPipeline:

    """

    Usage:
        folder_name = 'openai_2024_06_26_13_26_43_732914' # ←成功といってもいいだろう！！
        # folder_name = 'openai_2024_06_19_13_50_35_467793'

        model_name = 'model250000.pt'
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
            "use_boundary": False,
            "use_boundary_attn": True,
        }
        diffusion, model = create_diffusion_and_transformer(in_channels=hyperparams["in_channels"],
                                            condition_channels=hyperparams["condition_channels"],
                                            model_channels=hyperparams["model_channels"],
                                            out_channels=hyperparams["out_channels"],
                                            dataset=hyperparams["dataset"],
                                            use_checkpoint=hyperparams["use_checkpoint"],
                                            use_unet=hyperparams["use_unet"],
                                            analog_bit=hyperparams["analog_bit"],
                                            use_boundary=hyperparams["use_boundary"],
                                            use_boundary_attn=hyperparams["use_boundary_attn"],)
        model_base_path = f'/work/floorplangen/ckpts/{folder_name}/{model_name}'
        # diffusion, model = create_diffusion_and_transformer(condition_channels=89, )
        # モデルロード
        # model.load_state_dict(th.load(model_base_path, map_location="cpu"), strict=False)
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        model.load_state_dict(th.load(model_base_path, map_location=device), strict=False)
        model.to(device)
        model.eval()

        from rplan_datautil import load_rplanhg_data
        # Dataloader
        batch_size =256
        analog_bit = False
        target_set = 8
        set_name = 'train' # とりあえずtrain
        # set_name = 'eval' # trainはメモリリークするので
        data = load_rplanhg_data(
                    batch_size=batch_size,
                    analog_bit=analog_bit,
                    target_set=target_set,
                    set_name=set_name,
        )
        eval_pipe = EvalPipeline(diffusion, model, data, folder_name, model_name)
        eval_pipe.inference()
        iou_result = eval_pipe.iou()
    """

    def __init__(self, diffusion, model, data, folder_name='openai_2024_06_26_13_26_43_732914', model_name='model250000.pt'):

        self.diffusion = diffusion
        self.model = model
        self.data = data
        self.device = dist_util.dev()

        model_base_path = f'/work/floorplangen/ckpts/{folder_name}/{model_name}'
        # モデルロード
        model.load_state_dict(th.load(model_base_path, map_location=self.device), strict=False)
        model.to(self.device)
        print(f'model loaded at {self.device}')
        model.eval()

    
    def inference(self, steps=3, gt=False):
        # 結果保持
        self.images_colors = []
        self.images_colors_gt = []
        self.model_kwargs_list = []
        # for i in tqdm(range(len(dataset) % batch_size)):
        for i in tqdm(range(steps), leave=False, position=0):
            sample_fn = self.diffusion.p_sample_loop
            data_sample, model_kwargs = next(self.data)
            data_sample = data_sample.to(self.device)
            model_kwargs = {k: v.to(self.device) if isinstance(v, th.Tensor) else v for k, v in model_kwargs.items()}
            self.model_kwargs_list.append(model_kwargs)

            sample = sample_fn(
                self.model,
                data_sample.shape,
                model_kwargs=model_kwargs
            )
            
            sample = sample.permute([0, 1, 3, 2])
            # [29, batch_size, 100, 2]
            # ground truth も確認する
            if gt:
                sample_gt = data_sample.unsqueeze(0)
                # [1, batch_size, 100, 2]
                sample_gt = sample_gt.permute([0, 1, 3, 2])
            

            # 最後のt=1000だけでOK.そうすることでt=1000の印字もなくなる
            images_color = vis_sample(sample[-1:], model_kwargs, gif=False)
            self.images_colors.append(images_color)
            if gt:
                images_color_gt = vis_sample(sample_gt, model_kwargs, gif=False)
                self.images_colors_gt.append(images_color_gt)
        
    def iou(self):
        # self.masks = []
        iou_mean_list = []
        for images_color, model_kwargs in zip(self.images_colors, self.model_kwargs_list):
            masks = []
            for image in images_color:
                mask =create_mask_from_image(image[0], target_color=(255, 255, 255))
                masks.append(mask)
            
            # ground truthの外壁境界のmask取得
            boundary_images = plot_interpolated_boundary(model_kwargs, fill='black')
            
            iou_list = []
            # IoU計算
            for mask, boundary_img in zip(masks, boundary_images):
                iou = calculate_iou(np.array(mask), np.array(boundary_img)[:, :, 0])
                iou_list.append(iou)
            iou_mean_list.append(np.array(iou_list).mean())
        
        return np.array(iou_mean_list).mean()