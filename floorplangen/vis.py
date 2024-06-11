import io
from tqdm import tqdm
import drawSvg as drawsvg
import PIL.Image as Image
import numpy as np
import cairosvg
import webcolors


ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8',
            6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 11: '#727171',
            13: '#785A67', 12: '#D3A2C7'}
def vis_sample(sample, model_kwargs, door_indices=[11, 12, 13]):

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
        images, images2, images3 = vis_sample(sample_gt, model_kwargs)

        plt.imshow(images2[0])
        plt.axis('off')
        plt.show()

    '''
    resolution = 256
    for i in tqdm(range(sample.shape[1])):
        images = []
        images2 = []
        images3 = []
        for k in range(sample.shape[0]):
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
            i = 0
            for j, point in (enumerate(sample[k][i])): # process each point [100, 2]
                if model_kwargs['src_key_padding_mask'][i][j] == 1:
                    continue
                point = point.cpu().numpy()
                if j==0:
                    poly = []
                if j>0 and (model_kwargs['room_indices'][i, j]!=model_kwargs['room_indices'][i, j-1]).any():
                    polys.append(poly)
                    types.append(c)
                    poly = []
                point = point/2 + 0.5
                point = point * resolution
                poly.append((point[0], point[1]))

                c = np.argmax(model_kwargs['room_types'][i][j-1].numpy())
            polys.append(poly)
            types.append(c)
        # ドア以外の描画
        for poly, c in zip(polys, types):
            # ドアは無視
            if c in door_indices or c==0:
                continue
            room_type = c
            c = webcolors.hex_to_rgb(ID_COLOR[c])
            print(poly)
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
            room_type = c
            c = webcolors.hex_to_rgb(ID_COLOR[c])
            draw_color.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='black', stroke_width=1))
            draw.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill='black', fill_opacity=0.0, stroke=webcolors.rgb_to_hex([int(x/2) for x in c]), stroke_width=0.5*(resolution/256)))
            draw2.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke=webcolors.rgb_to_hex([int(x/2) for x in c]), stroke_width=0.5*(resolution/256)))
            for corner in poly:
                draw.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
                draw3.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
        images.append(Image.open(io.BytesIO(cairosvg.svg2png(draw.asSvg()))))
        images2.append(Image.open(io.BytesIO(cairosvg.svg2png(draw2.asSvg()))))
        images3.append(Image.open(io.BytesIO(cairosvg.svg2png(draw3.asSvg()))))
        
        return images, images2, images3