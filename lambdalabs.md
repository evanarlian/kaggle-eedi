# lambdalabs
My personal flow.

Make new SSH key (just once) and copy to every lambda instances to allow github access. Make sure to add SSH key to github as well.
```bash
# just once
ssh-keygen -t ed25519 -f ~/.ssh/lambdalabs -N ""

LAMBDA_IP=...
scp ~/.ssh/lambdalabs ubuntu@$LAMBDA_IP:~/.ssh/id_ed25519
scp -r ~/.kaggle ubuntu@$LAMBDA_IP:~
scp ~/.tmux.conf ubuntu@$LAMBDA_IP:~
```

On lambda vm, use tmux now to prevent interruptions.
```bash
git clone git@github.com:evanarlian/kaggle-eedi.git
cd kaggle-eedi
# open vscode remote, super slick
# manually fill .env first, or copy from local machine
export $(cat .env | xargs)
```

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```

```bash
conda create -n kaggle_eedi python=3.11 -y
conda activate kaggle_eedi
pip install -e .
```

```bash
accelerate config
ln -s ~/.cache/huggingface/accelerate/default_config.yaml .
```

```bash
git config --global credential.helper store
huggingface-cli login --token $HF_TOKEN --add-to-git-credential
wandb login $WANDB_TOKEN
```
