# kaggle-eedi
Eedi - Mining Misconceptions in Mathematics. See my kaggle [solution](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551659).


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
**Note**: this is only done once, you can download paraphrased data [here](https://www.kaggle.com/datasets/evanarlian/eedi-paraphrased).

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
**Note**: this is only done once, you can download paraphrased data [here](https://www.kaggle.com/datasets/evanarlian/eedi-synthetic).

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
