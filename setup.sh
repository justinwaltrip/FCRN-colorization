#!/bin/bash

# create venv
sudo apt update
sudo apt install -y python3.10-venv
python3 -m venv .env
source .env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# install os packages 
sudo apt install -y gcc make

# install nvidia driver
wget https://us.download.nvidia.com/tesla/460.106.00/NVIDIA-Linux-x86_64-460.106.00.run
sudo bash NVIDIA-Linux-x86_64-460.106.00.run
