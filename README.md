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

# notes and ideas:
* entropix
* translate to Chinese (most models are from china, both llm and embedding)
* use kaggle docker image and vscode run inside
* QWEN has MATH version, actually 2, just math and math-instruct. Super interesting
* maybe preprocess the latex to just normal math?
* paraphrase the question and misconception
* maybe we can use llm's entropy just like my experiment back then using meta galactica (this will consume a lot of time tho, beware)


# todo
* (DONE) Just emb untrained
* Paraphrase
* Translate
* Train emb
* Qwen -> bge -> top 25
* Bge top 5 -> qwen get top1 (but how do i get all 25?)
* Bge top 25 -> qwen entropy on all 25, rerank top 25 (this is kinda hard because we are forcing the llm not to think, using this way)
* Entropix
* ColBert style retrieval?
