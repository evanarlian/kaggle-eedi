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
1. Find out how bge m3 works. Is that really from hidden states? How does it handle padding?
2. Train bge m3 from those datasets, focusing on "given subject, cosntruct, question (or we can skip these?), what misconceptions might occur. This is easier than retrieving fine grained misconceptions with answer". I need to go crazy on these, make new dataset, paraphrase etc. Just for training. Step 2) is for the initial gathering. When using borda count, make sure for the emb model to souce from different place. Eg bge (BAAI) m3 for colbert and gte (Alibaba) for sent embedding.
3. Feed those initial gathered stuff from step2, along with cosntruct, subject, question, corrent ans, wrong ans, and feed maybe some of them to Qwen (or qwen math). This is the double retrieval trick from 0.36 kaggle score notebook. Can try dspy or entropix.
4. Use similarity again to map qwen output to narrowed down.

masterplan tl;dr:
We need to solve 2 hard problems. The first one is coarse search. We need to have as high as possible map@k. Use borda count
The second one is llm choice. For extra juice, we need entropix

colbert
* try qwen and colbert. First, let qwen generate simple answer first, and then you basically work on new dataset. 
* author suggests that we pad the query token to 32 (with [MASK]) or truncate. padding with [MASK] can force the model to "reweigh" or "rethink" the embedding to better fit retrieval task. See ablation study 4.4 (in colbert paper) for this! Some notable interesting thing is, late interaction computation is ofter not accompanied with attn mask, wow!  I think this is the [MASK] reweighing in action. 
* if shit does not work, time to use RAGatouile
* how did people train colpali
* llm2vec might be useful for training qwen math to understand better????????? this will take a lot of time tho

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
