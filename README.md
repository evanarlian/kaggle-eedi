# kaggle-eedi
Eedi - Mining Misconceptions in Mathematics


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

## synthetic data generation
Use openai gpt-4o to generate synthetic data increase dataset size. Some details:
* For misconceptions present in train, use 1-shot from the actual row in train set, then let the model generate 3 things: question, correct answer, and wrong answer.
* For misconceptions not present in train, use 2-shot hardcoded in the prompt, then let the model generate 5 things: subject, construct, question, correct answer, and wrong answer.
* Misconceptions are not changed at all, i.e. misconceptions were not generated.
* I did some light skimming and the there are quite many incorrect result. This might be because I did not use reasoning during text generation (expensive and slow).
* Synthetic generation costs about $30
* There are around 31500 synthetic rows and 4300 original (non synthetic) rows.
```bash
python eedi/generate_synthetic.py --dataset-dir=data 
```
**Note**: this is only done once, you can download paraphrased data [here](https://www.kaggle.com/datasets/evanarlian/eedi-synthetic). TODO make public later.

## finetune embedding model
Finetune embedding model with hard negative mining. First, download paraphrased dataset.
```bash
./scripts/download_paraphrased_data.sh
./scripts/download_synthetic_data.sh
```
Edit training script and run it.
```bash
./scripts/train.sh
```


# todo
* fix notebooks and stray py files (unimportant, do it last) 
* learn to use vast ai, they are so much cheaper, just use the datacenter one if not sure (in later comps)
* try saving each models (after eval) to huggingface, just for safeguard in case of overfitting
* wandb log args only log on rank 0. Log global batch size
* how to do eval only on rank 0 deepspeed?
* is there a way to avoid stateful config? (accelerate config is stateful and i prefer the explicit way)
* retrieval
  * use 14B dude solution, must change how complete question is structured
* rerank
  * Chinese borda count
  * soft borda count using logits?
  * dspy on awq vllm (i think this is fine since vllm has openai api), i think i need to rent lambda again to optimize dspy
  * Find more creative way to do test time compute


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