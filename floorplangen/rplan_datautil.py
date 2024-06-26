import math
import random
import torch as th

from PIL import Image, ImageDraw
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from glob import glob
import json
import os
import cv2 as cv
from tqdm import tqdm
from shapely import geometry as gm
from shapely.ops import unary_union
from collections import defaultdict
from glob import glob
from scipy.interpolate import interp1d
import copy

def load_rplanhg_data(
    batch_size,
    analog_bit,
    target_set = 8,
    set_name = 'train',
):
    """
    For a dataset, create a generator over (shapes, kwargs) pairs.
    """
    print(f"loading {set_name} of target set {target_set}")
    deterministic = False if set_name=='train' else True
    dataset = RPlanhgDataset(set_name, analog_bit, target_set)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False
        )
    while True:
        yield from loader

def make_non_manhattan(poly, polygon, house_poly):
    dist = abs(poly[2]-poly[0])
    direction = np.argmin(dist)
    center = poly.mean(0)
    min = poly.min(0)
    max = poly.max(0)

    tmp = np.random.randint(3, 7)
    new_min_y = center[1]-(max[1]-min[1])/tmp
    new_max_y = center[1]+(max[1]-min[1])/tmp
    if center[0]<128:
        new_min_x = min[0]-(max[0]-min[0])/np.random.randint(2,5)
        new_max_x = center[0]
        poly1=[[min[0], min[1]], [new_min_x, new_min_y], [new_min_x, new_max_y], [min[0], max[1]], [max[0], max[1]], [max[0], min[1]]]
    else:
        new_min_x = center[0]
        new_max_x = max[0]+(max[0]-min[0])/np.random.randint(2,5)
        poly1=[[min[0], min[1]], [min[0], max[1]], [max[0], max[1]], [new_max_x, new_max_y], [new_max_x, new_min_y], [max[0], min[1]]]

    new_min_x = center[0]-(max[0]-min[0])/tmp
    new_max_x = center[0]+(max[0]-min[0])/tmp
    if center[1]<128:
        new_min_y = min[1]-(max[1]-min[1])/np.random.randint(2,5)
        new_max_y = center[1]
        poly2=[[min[0], min[1]], [min[0], max[1]], [max[0], max[1]], [max[0], min[1]], [new_max_x, new_min_y], [new_min_x, new_min_y]]
    else:
        new_min_y = center[1]
        new_max_y = max[1]+(max[1]-min[1])/np.random.randint(2,5)
        poly2=[[min[0], min[1]], [min[0], max[1]], [new_min_x, new_max_y], [new_max_x, new_max_y], [max[0], max[1]], [max[0], min[1]]]
    p1 = gm.Polygon(poly1)
    iou1 = house_poly.intersection(p1).area/ p1.area
    p2 = gm.Polygon(poly2)
    iou2 = house_poly.intersection(p2).area/ p2.area
    if iou1>0.9 and iou2>0.9:
        return poly
    if iou1<iou2:
        return poly1
    else:
        return poly2

get_bin = lambda x, z: [int(y) for y in format(x, 'b').zfill(z)]
get_one_hot = lambda x, z: np.eye(z)[x]
class RPlanhgDataset(Dataset):
    def __init__(self, set_name, analog_bit, target_set, non_manhattan=False):
        super().__init__()
        base_dir = '../rplan_json'
        self.non_manhattan = non_manhattan
        self.set_name = set_name
        self.analog_bit = analog_bit
        self.target_set = target_set
        self.subgraphs = []
        self.org_graphs = []
        self.org_houses = []
        self.org_boundary_coords = []
        max_num_points = 100
        self.filenames = []
        # 追加
        if not os.path.exists('processed_rplan'):
            os.makedirs('processed_rplan')
        if self.set_name == 'eval':
            cnumber_dist = np.load(f'processed_rplan/rplan_train_{target_set}_cndist.npz', allow_pickle=True)['cnumber_dist'].item()
        if os.path.exists(f'processed_rplan/rplan_{set_name}_{target_set}.npz'):
            data = np.load(f'processed_rplan/rplan_{set_name}_{target_set}.npz', allow_pickle=True)
            self.graphs = data['graphs']
            self.houses = data['houses']
            self.door_masks = data['door_masks']
            self.self_masks = data['self_masks']
            self.gen_masks = data['gen_masks']
            self.num_coords = 2
            self.max_num_points = max_num_points
            self.filenames = data['filenames']
            self.boundary_points = data['boundary_points']
            self.interpolated_boundary_points = data['interpolated_boundary_points']
            cnumber_dist = np.load(f'processed_rplan/rplan_train_{target_set}_cndist.npz', allow_pickle=True)['cnumber_dist'].item()
            if self.set_name == 'eval':
                data = np.load(f'processed_rplan/rplan_{set_name}_{target_set}_syn.npz', allow_pickle=True)
                self.syn_graphs = data['graphs']
                self.syn_houses = data['houses']
                self.syn_door_masks = data['door_masks']
                self.syn_self_masks = data['self_masks']
                self.syn_gen_masks = data['gen_masks']
                self.filenames = data['filenames']
        else:
            # with open(f'{base_dir}/list.txt') as f:
                # lines = f.readlines()
            # cnt=0
            json_files = glob(f'{base_dir}/*.json')
            cnt = 0
            self.filenames = []
            for file_name in tqdm(json_files):
                self.current_file = file_name
                cnt=cnt+1
                # file_name = f'{base_dir}/{line[:-1]}'
                rms_type, fp_eds,rms_bbs,eds_to_rms,boundary_coords=reader(file_name) 
                fp_size = len([x for x in rms_type if x != 15 and x != 17])
                if self.set_name=='train' and fp_size == target_set:
                        continue
                if self.set_name=='eval' and fp_size != target_set:
                        continue
                a = [rms_type, rms_bbs, fp_eds, eds_to_rms,boundary_coords]
                self.filenames.append(int(os.path.basename(file_name).split('.')[0])) # ファイル名も保存する.文字列だとnumpyで保存できないのでintで(なのでファイル名は123.jsonの形を想定)
                self.subgraphs.append(a)
            for graph in tqdm(self.subgraphs):
                rms_type = graph[0]
                rms_bbs = graph[1]
                fp_eds = graph[2]
                eds_to_rms= graph[3]
                boundary_coords=graph[4]
                rms_bbs = np.array(rms_bbs)
                fp_eds = np.array(fp_eds)
                boundary_coords = np.array(boundary_coords)

                # extract boundary box and centralize
                # さらにやる意味ある．．．？

                tl = np.min(rms_bbs[:, :2], 0)
                br = np.max(rms_bbs[:, 2:], 0)
                shift = (tl+br)/2.0 - 0.5
                rms_bbs[:, :2] -= shift
                rms_bbs[:, 2:] -= shift
                fp_eds[:, :2] -= shift
                fp_eds[:, 2:] -= shift
                boundary_coords -= shift
                tl -= shift
                br -= shift

                # build input graph
                graph_nodes, graph_edges, rooms_mks = self.build_graph(rms_type, fp_eds, eds_to_rms)

                house = []
                for room_mask, room_type in zip(rooms_mks, graph_nodes):
                    room_mask = room_mask.astype(np.uint8)
                    room_mask = cv.resize(room_mask, (256, 256), interpolation = cv.INTER_AREA)
                    # 左上から左回りに境界の座標を取得(x,y)
                    contours, _ = cv.findContours(room_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                    contours = contours[0] # タプルの形になっているので，contours情報が入っている一つ目の要素だけ取得
                    # この時点で(4点なら)[4, 1, 2]の形になっているので[4, 2]の形にする(4は4点，2はx,y)
                    house.append([contours[:,0,:], room_type])
                self.org_graphs.append(graph_edges)
                # print('graph_edges.shape', graph_edges.shape)
                self.org_houses.append(house)
                self.org_boundary_coords.append(boundary_coords)
            # この時点で全データセットがorg_graphsに[node1, edge, node2]の形で多重リストで全間取り図のそれぞれのgraph関係が
            # org_housesに全間取りの境界情報[[x, y]の線分nparray, 部屋タイプ]がそれぞれの間取り分格納される
            houses = []
            door_masks = []
            self_masks = []
            gen_masks = []
            graphs = []
            boundary_points = []
            interpolated_boundary_points = []
            if self.set_name=='train':
                cnumber_dist = defaultdict(list)

            if self.non_manhattan:
                for h, graph in tqdm(zip(self.org_houses, self.org_graphs), desc='processing dataset'):
                    # Generating non-manhattan Balconies
                    tmp = []
                    for i, room in enumerate(h):
                        if room[1]>10:
                            continue
                        if len(room[0])!=4: 
                            continue
                        if np.random.randint(2):
                            continue
                        poly = gm.Polygon(room[0])
                        house_polygon = unary_union([gm.Polygon(room[0]) for room in h])
                        room[0] = make_non_manhattan(room[0], poly, house_polygon)
            for h, graph, boundary in tqdm(zip(self.org_houses, self.org_graphs, self.org_boundary_coords), desc='processing each house'):
                house = []
                corner_bounds = []
                num_points = 0
                skip_house = False  # 追加：この家データをスキップするかどうかのフラグ
                for i, room in enumerate(h):
                    # 部屋のタイプが10よりも大きい
                    if room[1]>10:
                        # 15->11
                        # 17->12
                        # 16->13
                        # に部屋タイプを変更
                        room[1] = {15:11, 17:12, 16:13}[room[1]]
                    # room[0]にはその部屋のedge情報が[[x1,y1], [x2,y2],...]とは言ってるが，それをnumpy arrayに変換
                    # さらに-0.5-0.5->[-1, 1]に正規化
                    room[0] = np.reshape(room[0], [len(room[0]), 2])/256. - 0.5 # [[x0,y0],[x1,y1],...,[x15,y15]] and map to 0-1 - > -0.5, 0.5
                    room[0] = room[0] * 2 # map to [-1, 1]
                    if self.set_name=='train':
                        # 部屋タイプのindexに，edgeの数をappend
                        # globalの部屋タイプにおけるedgeの数の統計を記録している(evalで使用)
                        cnumber_dist[room[1]].append(len(room[0]))
                    # Adding conditions
                    num_room_corners = len(room[0])
                    if num_room_corners >= 32: # 32以上だとエラーになってしまうのでskipするようにする
                        skip_house = True  # スキップフラグを立てる
                        break  # ループを抜けてこの家データをスキップ
                    # それぞれのedgeにつきroom typeをone hot (25次元) rtype.shape=(線の数, 25)(基本どこも同じところが1になるはず)
                    rtype = np.repeat(np.array([get_one_hot(room[1], 25)]), num_room_corners, 0)
                    # それぞれの部屋番号をone hot (32次元) room_index.shape=(線の数, 32)(基本どこも同じところが1になるはず)
                    room_index = np.repeat(np.array([get_one_hot(len(house)+1, 32)]), num_room_corners, 0)
                    # それぞれのedgeをone hot (最初のedgeは1次元目，次は2次元目,,,,)のようにone hotになる．shapeは(線の数, 32)
                    corner_index = np.array([get_one_hot(x, 32) for x in range(num_room_corners)])
                    # Src_key_padding_mask
                    # 全ての角に対してpadding maskを作成し，次元を拡張する．（attentionで使用する？)
                    # 多分だけど，self Attentionで，maskをするのでその時に使う？
                    padding_mask = np.repeat(1, num_room_corners)
                    padding_mask = np.expand_dims(padding_mask, 1)
                    # ここでのpadding maskのshapeは(角の数,1)
                    # Generating corner bounds for attention masks
                    # それぞれの角のつながりをリスト化してる
                    # 例:connections = array([[67, 68],
                                           # [68, 69],
                                           # [69, 70],
                                           # [70, 67]])
                    connections = np.array([[i,(i+1)%num_room_corners] for i in range(num_room_corners)])
                    connections += num_points
                    # corner_boundsには各部屋の各index範囲を記録する
                    # [[0, 4],
                    #  [4, 8],
                    #  [8, 12], ...]
                    # みたいな感じ
                    corner_bounds.append([num_points, num_points+num_room_corners])
                    # その部屋の角の数を更新(ここでは各部屋に対して処理をしてるので)
                    num_points += num_room_corners
                    # これらを全て一つのnp arrayに保存
                    room = np.concatenate((room[0], rtype, corner_index, room_index, padding_mask, connections), 1)
                    # houseのリストに格納していく
                    house.append(room)
                if skip_house:
                    continue
                house_layouts = np.concatenate(house, 0)
                # 角が多すぎる家は除外
                if len(house_layouts)>max_num_points:
                    continue
                # house_layoutsを[max_num_points, 94]の形にする(94は，前段のroomのconcatで全ての特徴ベクトルをconcatした際の次元数)
                padding = np.zeros((max_num_points-len(house_layouts), 94))
                # おそらくモデルの入力サイズが100x100なので，入力サイズを固定するために100x100にして，データの有効性をmaskとして渡すんだと思う
                gen_mask = np.ones((max_num_points, max_num_points))
                gen_mask[:len(house_layouts), :len(house_layouts)] = 0
                house_layouts = np.concatenate((house_layouts, padding), 0)

                door_mask = np.ones((max_num_points, max_num_points))
                self_mask = np.ones((max_num_points, max_num_points))
                for i in range(len(corner_bounds)):
                    for j in range(len(corner_bounds)):
                        if i==j:
                            # i, jが同じ部屋のならself_maskを0
                            self_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[j][0]:corner_bounds[j][1]] = 0
                        elif any(np.equal([i, 1, j], graph).all(1)) or any(np.equal([j, 1, i], graph).all(1)):
                            # iとjが接続されているならdoor_maskを0
                            door_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[j][0]:corner_bounds[j][1]] = 0
                houses.append(house_layouts)
                door_masks.append(door_mask)
                self_masks.append(self_mask)
                gen_masks.append(gen_mask)
                graphs.append(graph)
                
                
                interpolated_boundary = interpolate_loop(boundary, num_points=max_num_points)
                boundary = (boundary * 2.0) - 1.0
                interpolated_boundary = (interpolated_boundary * 2.0) - 1.0

                # boundary_coordsをパディングして追加. condはdictだけど，それぞれのcondの値はバッチごとにstackしたnparrayなので長さを揃える必要がある
                padded_boundary = np.zeros((max_num_points, 2))
                # # boundary_coordsを0〜1->1〜1に変換(他の座標に合わせる．多分これでいいはず)
                # boundary = (boundary * 2.0) - 1.0
                padded_boundary[:len(boundary)] = boundary

                boundary_points.append(padded_boundary)
                interpolated_boundary_points.append(interpolated_boundary)
            self.max_num_points = max_num_points
            self.houses = houses
            self.door_masks = door_masks
            self.self_masks = self_masks
            self.gen_masks = gen_masks
            self.num_coords = 2
            self.graphs = graphs
            self.boundary_points = boundary_points
            self.interpolated_boundary_points = interpolated_boundary_points

            print(len(self.filenames), len(self.graphs))
            # 他のデータ同様100次元に合わせる必要がある(バージョンの問題だったから不要かも？)
            self.filenames_expand = np.array([np.full((max_num_points,1), filename, int) for filename in self.filenames])
            # print(self.filenames_expand[0].shape, self.graphs[0].shape, self.door_masks[0].shape, self.houses[0].shape, self.self_masks[0].shape, self.gen_masks[0].shape)
            # Numpyのバージョンが新しいと，以下のコードで以下のエラーがでる
            '''
            setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (412,) + inhomogeneous part.
            '''
            # なので，本コードを実行する際には!pip install numpy==1.21.5をしておくこと．このエラーの原因は，self.graphsのshapeが他のnumpyのshapeと異なるから(他は100次元にpaddingされてるが，graph)
            np.savez_compressed(f'processed_rplan/rplan_{set_name}_{target_set}', graphs=self.graphs, houses=self.houses,
                    door_masks=self.door_masks, self_masks=self.self_masks, gen_masks=self.gen_masks, filenames=self.filenames_expand, boundary_points=self.boundary_points, interpolated_boundary_points=self.interpolated_boundary_points)
            if self.set_name=='train':
                np.savez_compressed(f'processed_rplan/rplan_{set_name}_{target_set}_cndist', cnumber_dist=cnumber_dist)
                # cnumber_distには，cornerのnumberのdistributionが 部屋のタイプindex:edgeの数

            if set_name=='eval':
                # 評価データでは，データセットの統計データをもとに部屋タイプに応じて角数をランダムサンプルして擬似データを作ってるみたい
                houses = []
                graphs = []
                door_masks = []
                self_masks = []
                gen_masks = []
                len_house_layouts = 0
                for h, graph in tqdm(zip(self.org_houses, self.org_graphs), desc='processing dataset'):
                    # h: それぞれの部屋の正規化された角の座標のリスト[[x1,y1],[x2,y2],[x3,y3],[x4,y4],type]のリスト(家単位)
                    # graph: それぞれの部屋と部屋のつながり情報[部屋id, 1, 部屋id]のリスト(家単位)
                    house = []
                    corner_bounds = []
                    num_points = 0
                    # 多分だが，cnumber_distはすでにset_name='train'を実行していて作っているという想定
                    # cnumber_distには，cornerのnumberのdistributionが 部屋のタイプindex:edgeの数のリスト　という形で入っている
                    # 例:{3: [4, 4],
                         #5: [4, 7]}
                    # それぞれの部屋のタイプに合わせて部屋の角の数をランダムサンプルしている（cnumber_distには部屋タイプに応じた角の数のリスト(データ全体の統計)が入っているので，どれか一つランダムに取ってきてる
                    num_room_corners_total = [cnumber_dist[room[1]][random.randint(0, len(cnumber_dist[room[1]])-1)] for room in h]
                    # max_num_points以下になるように何度も実施する
                    while np.sum(num_room_corners_total)>=max_num_points:
                        num_room_corners_total = [cnumber_dist[room[1]][random.randint(0, len(cnumber_dist[room[1]])-1)] for room in h]

                    for i, room in enumerate(h):
                        # Adding conditions
                        num_room_corners = num_room_corners_total[i]
                        rtype = np.repeat(np.array([get_one_hot(room[1], 25)]), num_room_corners, 0)
                        room_index = np.repeat(np.array([get_one_hot(len(house)+1, 32)]), num_room_corners, 0)
                        corner_index = np.array([get_one_hot(x, 32) for x in range(num_room_corners)])
                        # Src_key_padding_mask
                        padding_mask = np.repeat(1, num_room_corners)
                        padding_mask = np.expand_dims(padding_mask, 1)
                        # Generating corner bounds for attention masks
                        connections = np.array([[i,(i+1)%num_room_corners] for i in range(num_room_corners)])
                        connections += num_points
                        corner_bounds.append([num_points, num_points+num_room_corners])
                        num_points += num_room_corners
                        # 擬似データなので，本来room[0]にその部屋の座標情報が入ってるけど，ここではzerosを入れてる
                        room = np.concatenate((np.zeros([num_room_corners, 2]), rtype, corner_index, room_index, padding_mask, connections), 1)
                        house.append(room)

                    house_layouts = np.concatenate(house, 0)
                    if np.sum([len(room[0]) for room in h])>max_num_points:
                        continue
                    # 前と同じようにpaddingをして100x94にして，maskを作ってる
                    padding = np.zeros((max_num_points-len(house_layouts), 94))
                    gen_mask = np.ones((max_num_points, max_num_points))
                    gen_mask[:len(house_layouts), :len(house_layouts)] = 0
                    house_layouts = np.concatenate((house_layouts, padding), 0)

                    door_mask = np.ones((max_num_points, max_num_points))
                    self_mask = np.ones((max_num_points, max_num_points))
                    for i, room in enumerate(h):
                        if room[1]==1:
                            living_room_index = i
                            break
                    for i in range(len(corner_bounds)):
                        is_connected = False
                        for j in range(len(corner_bounds)):
                            if i==j:
                                self_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[j][0]:corner_bounds[j][1]] = 0
                            elif any(np.equal([i, 1, j], graph).all(1)) or any(np.equal([j, 1, i], graph).all(1)):
                                door_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[j][0]:corner_bounds[j][1]] = 0
                                is_connected = True
                        if not is_connected:
                            door_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[living_room_index][0]:corner_bounds[living_room_index][1]] = 0

                    houses.append(house_layouts)
                    door_masks.append(door_mask)
                    self_masks.append(self_mask)
                    gen_masks.append(gen_mask)
                    graphs.append(graph)
                self.syn_houses = houses
                self.syn_door_masks = door_masks
                self.syn_self_masks = self_masks
                self.syn_gen_masks = gen_masks
                self.syn_graphs = graphs
                np.savez_compressed(f'processed_rplan/rplan_{set_name}_{target_set}_syn', graphs=self.syn_graphs, houses=self.syn_houses,
                        door_masks=self.syn_door_masks, self_masks=self.syn_self_masks, gen_masks=self.syn_gen_masks)

    def __len__(self):
        return len(self.houses)

    def __getitem__(self, idx):
        # idx = int(idx//20)
        arr = self.houses[idx][:, :self.num_coords]
        graph = np.concatenate((self.graphs[idx], np.zeros([200-len(self.graphs[idx]), 3])), 0)

        cond = {
                'door_mask': self.door_masks[idx],
                'self_mask': self.self_masks[idx],
                'gen_mask': self.gen_masks[idx],
                'room_types': self.houses[idx][:, self.num_coords:self.num_coords+25],
                'corner_indices': self.houses[idx][:, self.num_coords+25:self.num_coords+57],
                'room_indices': self.houses[idx][:, self.num_coords+57:self.num_coords+89],
                'src_key_padding_mask': 1-self.houses[idx][:, self.num_coords+89],
                'connections': self.houses[idx][:, self.num_coords+90:self.num_coords+92],
                'graph': graph,
                'filename': self.filenames[idx], # どのファイルから来ているかわかる様に追加,
                'boundary_points': self.boundary_points[idx],
                'interpolated_boundary_points': self.interpolated_boundary_points[idx]
                }
        # print(self.filenames[idx])
        if self.set_name == 'eval':
            syn_graph = np.concatenate((self.syn_graphs[idx], np.zeros([200-len(self.syn_graphs[idx]), 3])), 0)
            assert (graph == syn_graph).all(), idx
            cond.update({
                'syn_door_mask': self.syn_door_masks[idx],
                'syn_self_mask': self.syn_self_masks[idx],
                'syn_gen_mask': self.syn_gen_masks[idx],
                'syn_room_types': self.syn_houses[idx][:, self.num_coords:self.num_coords+25],
                'syn_corner_indices': self.syn_houses[idx][:, self.num_coords+25:self.num_coords+57],
                'syn_room_indices': self.syn_houses[idx][:, self.num_coords+57:self.num_coords+89],
                'syn_src_key_padding_mask': 1-self.syn_houses[idx][:, self.num_coords+89],
                'syn_connections': self.syn_houses[idx][:, self.num_coords+90:self.num_coords+92],
                'syn_graph': syn_graph,
                })
        if self.set_name == 'train':
            #### Random Rotate
            rotation = random.randint(0,3)
            if rotation == 1:
                arr[:, [0, 1]] = arr[:, [1, 0]]
                arr[:, 0] = -arr[:, 0]
            elif rotation == 2:
                arr[:, [0, 1]] = -arr[:, [1, 0]]
            elif rotation == 3:
                arr[:, [0, 1]] = arr[:, [1, 0]]
                arr[:, 1] = -arr[:, 1]

            ## To generate any rotation uncomment this

            # if self.non_manhattan:
                # theta = random.random()*np.pi/2
                # rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                             # [np.sin(theta), np.cos(theta), 0]])
                # arr = np.matmul(arr,rot_mat)[:,:2]

            # Random Scale
            # arr = arr * np.random.normal(1., .5)

            # Random Shift
            # arr[:, 0] = arr[:, 0] + np.random.normal(0., .1)
            # arr[:, 1] = arr[:, 1] + np.random.normal(0., .1)

        if not self.analog_bit:
            arr = np.transpose(arr, [1, 0])
            return arr.astype(float), cond
        else:
            ONE_HOT_RES = 256
            arr_onehot = np.zeros((ONE_HOT_RES*2, arr.shape[1])) - 1
            xs = ((arr[:, 0]+1)*(ONE_HOT_RES/2)).astype(int)
            ys = ((arr[:, 1]+1)*(ONE_HOT_RES/2)).astype(int)
            xs = np.array([get_bin(x, 8) for x in xs])
            ys = np.array([get_bin(x, 8) for x in ys])
            arr_onehot = np.concatenate([xs, ys], 1)
            arr_onehot = np.transpose(arr_onehot, [1, 0])
            arr_onehot[arr_onehot==0] = -1
            return arr_onehot.astype(float), cond

    def make_sequence(self, edges):
        """
    グラフの接続性を追従してエッジリストから頂点のシーケンスを作成します。

    パラメータ:
        edges (list): エッジのリストで、各エッジは4つの要素(x1, y1, x2, y2)を持つタプルであり、
                      2つの接続された頂点を表します。

    戻り値:
        list: 接続されたパスを形成する頂点のシーケンスのリスト。

    注意:
        この関数はエッジのリストを処理して連続する頂点のシーケンスを形成します。訪問済みのエッジと頂点を追跡し、
        効率的にパスを作成します。ループが検出された場合、現在のシーケンスを保存し、残りのエッジに対して処理を繰り返します。
    """
        polys = []
        v_curr = tuple(edges[0][:2])
        e_ind_curr = 0
        e_visited = [0]
        seq_tracker = [v_curr]
        find_next = False
        while len(e_visited) < len(edges):
            if find_next == False:
                if v_curr == tuple(edges[e_ind_curr][2:]):
                    v_curr = tuple(edges[e_ind_curr][:2])
                else:
                    v_curr = tuple(edges[e_ind_curr][2:])
                find_next = not find_next 
            else:
                # look for next edge
                for k, e in enumerate(edges):
                    if k not in e_visited:
                        if (v_curr == tuple(e[:2])):
                            v_curr = tuple(e[2:])
                            e_ind_curr = k
                            e_visited.append(k)
                            break
                        elif (v_curr == tuple(e[2:])):
                            v_curr = tuple(e[:2])
                            e_ind_curr = k
                            e_visited.append(k)
                            break

            # extract next sequence
            if v_curr == seq_tracker[-1]:
                polys.append(seq_tracker)
                for k, e in enumerate(edges):
                    if k not in e_visited:
                        v_curr = tuple(edges[0][:2])
                        seq_tracker = [v_curr]
                        find_next = False
                        e_ind_curr = k
                        e_visited.append(k)
                        break
            else:
                seq_tracker.append(v_curr)
        polys.append(seq_tracker)

        return polys

    def build_graph(self, rms_type, fp_eds, eds_to_rms, out_size=64):
        """
    部屋のタイプとエッジマッピングからグラフ構造を構築し、部屋のマスクを生成します。

    パラメータ:
        rms_type (list): 整数によって表される部屋のタイプのリスト。
        fp_eds (list): フロアプランのエッジのリスト。各要素はエッジの頂点を記述するタプルです。
        eds_to_rms (list): 部屋間の接続を示す、部屋にマッピングされたエッジのリスト。
        out_size (int, optional): 部屋のマスクの出力サイズ。デフォルトは64です。

    戻り値:
        tuple: 以下を含むタプル:
            - nodes (numpy.array): 部屋タイプの配列。
            - triples (numpy.array): グラフのエッジを表すトリプルの配列。各トリプルは[node1, edge, node2]です。
            - rms_masks (numpy.array): 部屋のマスクの配列。各マスクは部屋の存在を示すバイナリイメージです。

    注意:
        この関数は、最初にグラフのエッジを表すトリプルのリストを構築し、隣接をチェックし、
        トレーニングセットに基づいて条件を適用します。その後、フロアプランデータから部屋のマスクを生成し、
        指定された出力サイズにリサイズし、部屋の識別での重なりを調整します。"""
        # create edges
        triples = []
        nodes = rms_type 
        # encode connections
        for k in range(len(nodes)):
            for l in range(len(nodes)):
                if l > k:
                    is_adjacent = any([True for e_map in eds_to_rms if (l in e_map) and (k in e_map)])
                    if is_adjacent:
                        if 'train' in self.set_name:
                            triples.append([k, 1, l])
                        else:
                            triples.append([k, 1, l])
                    else:
                        if 'train' in self.set_name:
                            triples.append([k, -1, l])
                        else:
                            triples.append([k, -1, l])
        # get rooms masks
        eds_to_rms_tmp = []
        for l in range(len(eds_to_rms)):                  
            eds_to_rms_tmp.append([eds_to_rms[l][0]])
        rms_masks = []
        im_size = 256
        fp_mk = np.zeros((out_size, out_size))
        for k in range(len(nodes)):
            # add rooms and doors
            eds = []
            for l, e_map in enumerate(eds_to_rms_tmp):
                if (k in e_map):
                    eds.append(l)
            # draw rooms
            rm_im = Image.new('L', (im_size, im_size))
            dr = ImageDraw.Draw(rm_im)
            # for eds_poly in [eds]:
            #     poly = self.make_sequence(np.array([fp_eds[l][:4] for l in eds_poly]))[0]
            #     poly = [(im_size*x, im_size*y) for x, y in poly]
            #     if len(poly) >= 2:
            #         dr.polygon(poly, fill='white')
            #     else:
            #         print("Empty room")
            #         exit(0)
            try:
                poly = self.make_sequence(np.array([fp_eds[l][:4] for l in eds]))[0]
                poly = [(im_size*x, im_size*y) for x, y in poly]
                if len(poly) >= 2:
                    dr.polygon(poly, fill='white')
                else:
                    print("Empty room")
                    exit(0)
            except Exception as e:
                print(f"Error processing file: {self.current_file}, Error: {str(e)}")
                continue
            rm_im = rm_im.resize((out_size, out_size))
            rm_arr = np.array(rm_im)
            inds = np.where(rm_arr>0)
            rm_arr[inds] = 1.0
            rms_masks.append(rm_arr)
            if rms_type[k] != 15 and rms_type[k] != 17:
                fp_mk[inds] = k+1
        # trick to remove overlap
        for k in range(len(nodes)):
            if rms_type[k] != 15 and rms_type[k] != 17:
                rm_arr = np.zeros((out_size, out_size))
                inds = np.where(fp_mk==k+1)
                rm_arr[inds] = 1.0
                rms_masks[k] = rm_arr
        # convert to array
        nodes = np.array(nodes)
        triples = np.array(triples)
        rms_masks = np.array(rms_masks)
        return nodes, triples, rms_masks

def is_adjacent(box_a, box_b, threshold=0.03):
    x0, y0, x1, y1 = box_a
    x2, y2, x3, y3 = box_b
    h1, h2 = x1-x0, x3-x2
    w1, w2 = y1-y0, y3-y2
    xc1, xc2 = (x0+x1)/2.0, (x2+x3)/2.0
    yc1, yc2 = (y0+y1)/2.0, (y2+y3)/2.0
    delta_x = np.abs(xc2-xc1) - (h1 + h2)/2.0
    delta_y = np.abs(yc2-yc1) - (w1 + w2)/2.0
    delta = max(delta_x, delta_y)
    return delta < threshold

def reader(filename):
    with open(filename) as f:
        info =json.load(f)
        rms_bbs=np.asarray(info['boxes'])
        fp_eds=info['edges']
        rms_type=info['room_type']
        eds_to_rms=info['ed_rm']
        boundary_coords = info['boundary_coords']
        s_r=0
        for rmk in range(len(rms_type)):
            if(rms_type[rmk]!=17):
                s_r=s_r+1   
        rms_bbs = np.array(rms_bbs)/256.0
        fp_eds = np.array(fp_eds)/256.0
        boundary_coords = np.array(boundary_coords)/256.0 
        fp_eds = fp_eds[:, :4]
        tl = np.min(rms_bbs[:, :2], 0) # top left 
        br = np.max(rms_bbs[:, 2:], 0) # bottom right
        shift = (tl+br)/2.0 - 0.5 
        # 実際に値をみたらshiftが0（tl+br=1）になっていて，既に正規化されていると思われる
        rms_bbs[:, :2] -= shift 
        rms_bbs[:, 2:] -= shift
        fp_eds[:, :2] -= shift
        fp_eds[:, 2:] -= shift
        # boundary_coords.shape = [num_edge, 2]
        boundary_coords[:, :2] -= shift
        # この時点で0~1に正規化されている想定
        tl -= shift
        br -= shift
        return rms_type,fp_eds,rms_bbs,eds_to_rms,boundary_coords
    
def interpolate_loop(coords, num_points=100):
    """
    Interpolate the given loop coordinates to a specified number of points.
    
    Usage:
        from rplan_datautil import reader
        from glob import glob
        base_dir = '../rplan_json'
        json_files = glob(f'{base_dir}/*.json')
        rms_type, fp_eds,rms_bbs,eds_to_rms,boundary_coords = reader(json_files[0])
        new_coords = interpolate_loop(boundary_coords, num_points=100)

        # Show the interpolated loop (optional)
        plt.figure(figsize=(6, 6))
        plt.plot(boundary_coords[:, 0], boundary_coords[:, 1], 'o-', label='Original')
        plt.plot(new_coords[:, 0], new_coords[:, 1], 'x-', label='Interpolated')
        plt.legend()
        plt.show()

    Parameters:
        coords (numpy array): Array of shape [N, 2] with the loop coordinates.
        num_points (int): The number of points to interpolate.

    Returns:
        numpy array: Interpolated loop coordinates.
    """
    # Remove padding (0, 0) and get unique points
    coords = coords[~np.all(coords == 0, axis=1)]
    # coords = np.unique(coords, axis=0)
    
    # Close the loop
    if not np.array_equal(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    
    # Calculate the cumulative distance along the loop
    distances = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    total_distance = cumulative_distances[-1]
    
    # Create an interpolator for x and y coordinates
    interp_func_x = interp1d(cumulative_distances, coords[:, 0], kind='linear')
    interp_func_y = interp1d(cumulative_distances, coords[:, 1], kind='linear')
    
    # Generate new points along the loop
    new_distances = np.linspace(0, total_distance, num_points)
    new_coords_x = interp_func_x(new_distances)
    new_coords_y = interp_func_y(new_distances)
    
    new_coords = np.stack([new_coords_x, new_coords_y], axis=-1)
    return new_coords

if __name__ == '__main__':
    dataset = RPlanhgDataset('train', False, 8)
