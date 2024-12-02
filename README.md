# kaggle-eedi
Eedi - Mining Misconceptions in Mathematics

# note: just realized that DDP is not what people use for large model training :(

# preparation
Make virtual env and install deps.
```bash
pip install -e .
```
Copy `.env.example` to `.env` and add openai key (only for paraphrase).

Download dataset.
```bash
./scripts/download_data.sh
```

# usage
## paraphrase
Use openai gpt-4o mini to paraphrase the questions and the miconceptions to increase dataset size. For each question and misconception, create 4 more paraphrase. Costs about $0.36
```bash
python eedi/paraphrase.py --dataset-dir=data
```
**Note**: this is only done once, you can download paraphrased data [here](https://www.kaggle.com/datasets/evanarlian/eedi-paraphrased). TODO make public later.

## finetune embedding model
Finetune embedding model with hard negative mining. First, download paraphrased dataset.
```bash
./scripts/download_paraphrased_data.sh
```
Edit training script and run it.
```bash
./scripts/train.sh
```

# todo
MASTERPLAN:
1. [DONE] paraphrase question and misconception. add new column (ai_created). push to kaggle dataset but private.
2. [DONE] find out about the llm model that does not require trust remote code. Nvidia nvembed v2 is super bad because it need to change the sentence transformer code.
3. finetune the model based on KDE(?) cup 1st winner code on kaggle. Ref: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/543519
4. after rerank, ask qwen math to select the most appropriate misconception
   1. fix notebooks and stray py files

# lingering problems:
gpu problems
* check raw hf (non sentence transformer mem usage forward backward), fp16, with lora and not with lora
* check the effect of not loading the model to gpu (automatically) during the first load
* MultipleNegativeLoss might influence gpu vram usage

dataset problem:
* how to change dataset mid training
* dataset proxy, add custom method to do iterative hn. Called from callback
* How to do this only from the rank0?
  

# notes
on models to choose:
* i chose simple model first like alibaba gte because it might work lol
* the next llm model to choose is the salesforce SFR, since that is already done by that chinese person. also SFR can work with sentence transformers without much code changes.

# todo
* wandb log args only log on rank 0. Log global batch size
* how to do eval only on rank 0 deepspeed?
* use grad accumulation for larger models
* try deepspeed offloading to see the memory difference
* is there a way to avoid stateful config? (accelerate config is stateful and i prefer the explicit way)
* solve 'key' keyerror during psuh to hub, make sure only rank0 does that. Tokenizer is not pushed!!!!
* review all device-related stuffs

# lambdalabs
My personal flow so

Make new SSH key (just once) and copy to every lambda instances to allow github access. Make sure to add SSH key to github as well.

TODO copy tmux from here too, dont do it from this repo!
```bash
LAMBDA_IP=104.171.202.136
ssh-keygen -t ed25519 -f ~/.ssh/lambdalabs -N ""
scp ~/.ssh/lambdalabs ubuntu@$LAMBDA_IP:~/.ssh/id_ed25519
scp -r ~/.kaggle ubuntu@$LAMBDA_IP:~
```

On lambda

```bash
git clone git@github.com:evanarlian/kaggle-eedi.git
cd kaggle-eedi
# manually fill .env first
export $(cat .env | xargs)
cp ./scripts/.tmux.conf ~/
```

Open vscode server

Copy .env from local to remote

on lambda again
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
```

logins
```bash
git config --global credential.helper store
huggingface-cli login --token $HF_TOKEN --add-to-git-credential
wandb login $WANDB_TOKEN
```