# ベースとなるDockerイメージを指定
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# 必要なパッケージをインストール
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 gcc gfortran libhdf5-dev libopenmpi-dev pkg-config libcairo2 libcairo2-dev libgirepository1.0-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# シンボリックリンクを作成
# RUN ln -s /usr/bin/python3 /usr/bin/python && \
#     ln -s /usr/bin/pip3 /usr/bin/pip

# ワーキングディレクトリの設定
WORKDIR /work

# requirements.txtをコピー
COPY requirements.txt /work/requirements.txt

# Python パッケージをインストール
RUN pip install --upgrade pip && \
    pip install -r /work/requirements.txt

# JupyterLabをデフォルトコマンドとして実行
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''"]