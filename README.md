# kaggle-eedi
Eedi - Mining Misconceptions in Mathematics

# usage
Download files.
```bash
./scripts/download_data.sh
```

# notes and ideas:
* entropix
* u-search (probably not needed, just bruteforce lol)
* BGE large
* translate to Chinese (most models are from china, both llm and embedding)
* train emb model. TODO find out how
* use kaggle docker image and vscode run inside
* how does 
* QWEN has MATH version, actually 2, just math and math-instruct. Super interesting
* maybe preprocess the latex to just normal math?
* paraphrase the question and misconception
* maybe we can use llm's entropy just like my experiment back then using meta galactica (this will consume a lot of time tho, beware)


# todo
* complete eda, fill up with extra knowledge:
    * test already equipped with correct answer, so this competition is only about finding the misconception, no need to answer the question
* download QWEN,
* download kaggle docker
