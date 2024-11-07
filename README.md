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
2. find out about the llm model that does not require trust remote code. Nvidia nvembed v2 is super bad because it need to change the sentence transformer code.
3. finetune the model based on KDE(?) cup 1st winner code on kaggle. Code is on github i believe.
4. after rerank, ask qwen math to select the most appropriate misconception
