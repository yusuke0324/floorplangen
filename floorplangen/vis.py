import io
import imageio
import IPython.display as display
from tqdm import tqdm
import drawSvg as drawsvg
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cairosvg
import webcolors


ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8',
            6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 11: '#727171',
            13: '#785A67', 12: '#D3A2C7'}

def _add_text_to_image(image, text, position=(10, 10), font_size=60, font_color="white"):
    try:
        # システムのフォントが見つからない場合、デフォルトフォントを使用
        font = ImageFont.truetype("arial", font_size)
    except IOError:
        font = ImageFont.load_default()
    images_with_text = []
    # for img in images:
    img_with_text = image.copy()
    draw = ImageDraw.Draw(img_with_text)
    draw.text(position, text, font=font, fill=font_color)
    return img_with_text

def create_gif(images):
    # 画像のリストからGIFを作成してメモリ内に保存
    with io.BytesIO() as gif_buffer:
        imageio.mimwrite(gif_buffer, images, format='GIF', fps=10, loop=1)
        gif_buffer.seek(0)
        # GIFを表示
        return display.Image(data=gif_buffer.read(), format='png')

def vis_sample(sample, model_kwargs, door_indices=[11, 12, 13], first_t=971, gif=True):

    '''
    example: 
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
        data_sample, model_kwards = next(data)
        sample_gt = data_sample.unsqueeze(0)
        sample_gt = sample_gt.permute([0, 1, 3, 2])
        images, images2, images3 = vis_sample(sample_gt, model_kwargs, gif=True)

        gif = images[0]
        display.display(gif)

    '''
    resolution = 256
    batch_images = []
    batch_images2 = []
    batch_images3 = []
    for i in tqdm(range(sample.shape[1])):
        # 各バッチを処理
        images = []
        images2 = []
        images3 = []
        for k in range(sample.shape[0]):
            # 各t:971~1000を処理
            t = k + first_t
            # init all drawings
            draw = drawsvg.Drawing(resolution, resolution, displayInline=True)
            # 背景を黒
            draw.append(drawsvg.Rectangle(0, 0, resolution, resolution, fill='black'))
            draw2 = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw2.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='black'))
            draw3 = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw3.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='black'))            
            draw_color = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw_color.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='white'))
            polys = []
            types = []
            # i = 0
            for j, point in (enumerate(sample[k][i])): # process each point [100, 2]
                if model_kwargs['src_key_padding_mask'][i][j] == 1:
                    continue
                point = point.detach().cpu().numpy() # 学習の途中結果をみたい場合もあるのでdetachしておく
                if j==0:
                    poly = []
                if j>0 and (model_kwargs['room_indices'][i, j]!=model_kwargs['room_indices'][i, j-1]).any():
                    polys.append(poly)
                    types.append(c)
                    poly = []
                point = point/2 + 0.5
                point = point * resolution
                poly.append((point[0], point[1]))

                c = np.argmax(model_kwargs['room_types'][i][j-1].cpu().numpy())
            polys.append(poly)
            types.append(c)
            # ドア以外の描画
            for poly, c in zip(polys, types):
                # ドアは無視    
                if c in door_indices or c==0:
                    continue
                # バージョンの問題かもだけど，なぜかcがnumpyだとTypeError: unhashable type: 'numpy.ndarray'がでるので
                # 整数にするcast
                c = int(c)
                room_type = c
                c = webcolors.hex_to_rgb(ID_COLOR[c])
                # 各点を描画(線)
                # 背景白, 線黒, type色でfill
                draw_color.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='black', stroke_width=1))
                # 背景黒, 線type色, 黒でfill
                draw.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill='black', fill_opacity=0.0, stroke=webcolors.rgb_to_hex([int(x/2) for x in c]), stroke_width=0.5*(resolution/256)))
                #背景黒, 線type色, type色でfill
                draw2.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke=webcolors.rgb_to_hex([int(x/2) for x in c]), stroke_width=0.5*(resolution/256)))
                for corner in poly:
                    # 各部屋の角を丸い点にして角をわかりやすくしてる
                    draw.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
                    draw3.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
            # ドアの描画
            for poly, c in zip(polys, types):
                if c not in door_indices:
                    continue
                c = int(c)
                room_type = c
                c = webcolors.hex_to_rgb(ID_COLOR[c])
                draw_color.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='black', stroke_width=1))
                draw.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill='black', fill_opacity=0.0, stroke=webcolors.rgb_to_hex([int(x/2) for x in c]), stroke_width=0.5*(resolution/256)))
                draw2.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke=webcolors.rgb_to_hex([int(x/2) for x in c]), stroke_width=0.5*(resolution/256)))
                for corner in poly:
                    draw.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
                    draw3.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
            image = Image.open(io.BytesIO(cairosvg.svg2png(draw.asSvg())))
            image2 = Image.open(io.BytesIO(cairosvg.svg2png(draw2.asSvg())))
            image3 = Image.open(io.BytesIO(cairosvg.svg2png(draw3.asSvg())))

            image_with_text = _add_text_to_image(image, f't={t}')
            image2_with_text = _add_text_to_image(image2, f't={t}')
            image3_with_text = _add_text_to_image(image3, f't={t}')

            images.append(image_with_text)
            images2.append(image2_with_text)
            images3.append(image3_with_text)

            if gif:
                images_gif = create_gif(images)
                images2_gif = create_gif(images2)
                images3_gif = create_gif(images3)

        if gif:
            batch_images.append(images_gif)
            batch_images2.append(images2_gif)
            batch_images3.append(images3_gif)
        else:
            batch_images.append(images)
            batch_images2.append(images2)
            batch_images3.append(images3)

        
        
    return batch_images, batch_images2, batch_images3