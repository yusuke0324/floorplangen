# ベースとなるDockerイメージを指定
FROM python:3.9-slim

# 必要なパッケージをインストール
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 gcc python3-dev gfortran libhdf5-dev libopenmpi-dev pkg-config libcairo2 libcairo2-dev libgirepository1.0-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ワーキングディレクトリの設定
WORKDIR /work

# requirements.txtをコピー
COPY requirements.txt /work/requirements.txt

# Python パッケージをインストール
RUN pip install --upgrade pip && \
    pip install -r /work/requirements.txt

# JupyterLabをデフォルトコマンドとして実行
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''"]