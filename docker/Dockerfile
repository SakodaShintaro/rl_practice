FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# 更新
RUN apt-get update

# 言語の設定
RUN apt-get install -y language-pack-ja-base language-pack-ja
ENV LANG='ja_JP.UTF-8'

# 必要なもののインストール
RUN apt-get install -y sudo
RUN apt-get install -y vim
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y git

RUN pip3 install gym
RUN pip3 install gym[classic_control]

# dotfileの導入
WORKDIR /root
RUN git clone https://github.com/SakodaShintaro/dotfiles && ./dotfiles/setup.sh
