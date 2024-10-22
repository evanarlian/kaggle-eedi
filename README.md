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
colbert
* finish implement tiling colbert score (incrementally build colbert score)
* author suggests that we pad the query token to 32 (with [MASK]) or truncate. padding with [MASK] can force the model to "reweigh" or "rethink" the embedding to better fit retrieval task. See ablation study 4.4 (in colbert paper) for this! Some notable interesting thing is, late interaction computation is ofter not accompanied with attn mask, wow!  I think this is the [MASK] reweighing in action. 

GOD tier idea: LLM2Vec with Qwen math + colbert. Finetunes are for last resprt!
* https://arxiv.org/pdf/2404.05961
* https://github.com/McGill-NLP/llm2vec
wait wait wait a minute, if colpali and colqwen so popular, that means they are llms right? no need for llm2vec?
* read colpali
* read colqwen
* my biggest fear is that question-answer pair is soo much different than misconception, so colbert wont work, need finetune afterall
* make my own colbert class bc this one is super slow prolly, no gpu support as well. Compare the result with reference, handle padding edge cases
* find math embedding. i think QWEN math if it can be used as vector, itll be good 

Prolly good:
* QWEN predict batch, find the predictions with misconceptions list with colbert
* hybrid search
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

Postprocess
* hungarian matching
* generalized assignment problem
* bipartite matching
* edmonds karp algo
* ford fulkerson algo
* python pulp, google or tools
