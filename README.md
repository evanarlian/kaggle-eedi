# kaggle-eedi
Eedi - Mining Misconceptions in Mathematics

# preparation
Make virtual env and install deps.
```bash
pip install .
```

Download dataset.
```bash
./scripts/download_data.sh
```

# usage
See EDA on `eda.ipynb`.

# todo
MASTERPLAN:
1. paraphrase question and misconception. add new column (ai_created). push to kaggle dataset but private.
2. find out about the llm model that does not require trust remote code. Nvidia nvembed v2 is super bad because it need to change the sentence transformer code.
3. finetune the model based on KDE(?) cup 1st winner code on kaggle. Code is on github i believe.
4. after rerank, ask qwen math to select the most appropriate misconception
