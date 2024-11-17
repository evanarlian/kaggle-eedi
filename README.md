# kaggle-eedi
Eedi - Mining Misconceptions in Mathematics

# preparation
Make virtual env and install deps.
```bash
pip install .
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
python paraphrase.py --dataset-dir=data
```

# todo
MASTERPLAN:
1. [DONE] paraphrase question and misconception. add new column (ai_created). push to kaggle dataset but private.
2. [DONE] find out about the llm model that does not require trust remote code. Nvidia nvembed v2 is super bad because it need to change the sentence transformer code.
3. finetune the model based on KDE(?) cup 1st winner code on kaggle. Ref: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/543519
4. after rerank, ask qwen math to select the most appropriate misconception


# notes
on models to choose:
* i chose simple model first like alibaba gte because it might work lol
* the next llm model to choose is the salesforce SFR, since that is already done by that chinese person. also SFR can work with sentence transformers without much code changes.

locked in:
* make evaluator
* extra data by prompting gpt, idk how to make a diverse but not too different dataset

ideas:
* start very simple by skipping paraphrased
* use paraphrased. play with n_negatives retrieved.
* then start iterative
* curriculum learning (needed?) for top k hard, first try 100, 50, 25, etc. Smallest should be 25.

differences
* sentence transformers' encode is the same as hf's model(**encoded) blabla on [CLS] token.
