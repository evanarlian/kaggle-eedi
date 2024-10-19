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
GOD tier idea: LLM2Vec with Qwen math + colbert. Finetunes are for last resprt!
* https://arxiv.org/pdf/2404.05961
* https://github.com/McGill-NLP/llm2vec
wait wait wait a minute, if colpali and colqwen so popular, that means they are llms right? no need for llm2vec?
* real colpali
* read colqwen
* my biggest fear is that question-answer pair is soo much different than misconception, so colbert wont work, need finetune afterall

Prolly good:
* QWEN has MATH version, actually 2, just math and math-instruct. Super interesting
* (DONE) Just emb untrained
* Train emb
* ColBert style retrieval?
* LLM2Vec
* Qwen -> bge -> top 25
* Bge top 5 -> qwen get top1 (but how do i get all 25?)
* Bge top 25 -> qwen entropy on all 25, rerank top 25 (this is kinda hard because we are forcing the llm not to think, using this way)
* Entropix

Gimmicky:
* Translate
* Paraphrase
