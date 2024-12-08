import json
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
from vllm import LLM, SamplingParams

from eedi.datasets import make_complete_query, make_nice_df


@dataclass
class Args:
    dataset_dir: Path
    top25_path: Path
    model_path: str


def generate_numbering_seq(k: int, kind: Literal["number", "alphabet"]) -> list[str]:
    # TODO this might not be needed idk since we might only use number
    if kind == "number":
        return [str(i) for i in range(1, k + 1)]
    elif kind == "alphabet":
        return [chr(ord("A") + i) for i in range(k)]
    assert False, f"unsupported kind: {kind}"


def make_llm_prompt_en(
    row: pd.Series,
    k: int,
    orig_mis: list[str],
) -> str:
    question = row["QuestionComplete"]
    top25_mis: list[int] = row["Top25Miscons"]  # type: ignore
    # my own prompt
    # TODO answer with number only
    template = "You are an elite mathematics teacher tasked to assess the student's understanding of math concepts. Below, you will be presented with: the math question, the correct answer, the wrong answer and {k} possible misconceptions that could have led to the mistake.\n\n{question}\n\nPossible Misconceptions\n{choices}\n\nSelect one misconception that leads to incorrect answer.\n\nAnswer: "
    numbered_mis_texts = []
    for i, iseq in enumerate(generate_numbering_seq(k, "number")):
        numbered_mis_texts.append(f"{iseq}. {orig_mis[top25_mis[i]]}")
    numbered_mis_texts = "\n".join(numbered_mis_texts)
    llm_prompt = template.format(k=k, question=question, choices=numbered_mis_texts)
    return llm_prompt


def make_llm_prompt_zh(
    row: pd.Series,
    k: int,
    orig_mis: list[str],
) -> str:
    question = row["QuestionComplete"]
    top25_mis: list[int] = row["Top25Miscons"]  # type: ignore
    # adapted from Qwen 2.5 math prompt for GaoKao Math QA (figure 10)
    # TODO rework on chinese
    template = "选择题: 以下是数学题、正确答案、错误答案，以及可能导致错误答案的 {k} 种常见误解。\n\n{question}\n\n{choices}\n\n是什么误解可能导致了错误答案？在选择答案之前，先给出一个非常简短的解释。将答案选项的字母用 <answer></answer> 包裹起来。\n\n解: "
    numbered_mis_texts = []
    for i, iseq in enumerate(generate_numbering_seq(k, "alphabet")):
        numbered_mis_texts.append(f"{iseq}. {orig_mis[top25_mis[i]]}")
    numbered_mis_texts = "\n".join(numbered_mis_texts)
    llm_prompt = template.format(k=k, question=question, choices=numbered_mis_texts)
    return llm_prompt


def main(args: Args):
    # load dataset
    df_mis = pd.read_csv(args.dataset_dir / "misconception_mapping.csv")
    orig_mis = df_mis["MisconceptionName"].tolist()
    assert len(orig_mis) == 2587
    df_test = pd.read_csv(args.dataset_dir / "test.csv")
    df_test = make_nice_df(df_test)
    df_test["QuestionComplete"] = df_test.apply(make_complete_query, axis=1)
    with open(args.top25_path, "r") as f:
        top25_miscons = json.load(f)
    df_test["Top25Miscons"] = top25_miscons

    # load vllm model and tokenizer
    llm = LLM(
        args.model_path,
        quantization="awq",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=5120,
        disable_log_stats=True,
    )
    tokenizer = llm.get_tokenizer()

    # rerank
    RERANK = 9  # bc i dont want to handle double token like 10, 11, ..., super messy
    df_test["PromptEn"] = df_test.apply(
        lambda row: make_llm_prompt_en(row, RERANK, orig_mis), axis=1
    )
    df_test["PromptZh"] = df_test.apply(
        lambda row: make_llm_prompt_zh(row, RERANK, orig_mis), axis=1
    )

    # TODO english only for now
    logits_processor = MultipleChoiceLogitsProcessor(
        tokenizer=tokenizer,  # type: ignore
        choices=generate_numbering_seq(RERANK, "number"),
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0,
        max_tokens=1,
        logits_processors=[logits_processor],
        logprobs=RERANK,
    )
    responses_en = llm.generate(df_test["PromptEn"].tolist(), sampling_params)

    # remap back to original (TODO FOR english, does not work for zh yet)
    all_reranked = []
    for resp, top25 in zip(responses_en, df_test["Top25Miscons"]):
        decoded_tokens = [
            logprob.decoded_token
            for logprob in resp.outputs[0].logprobs[0].values()  # type: ignore
        ]
        # map back to 0-based int (e.g. ["1", "3", "4"] -> [0, 2, 3])
        indices = [int(d) - 1 for d in decoded_tokens]  # type: ignore
        # rerank the first 9 items from 25
        reranked = np.array(top25[:RERANK])[indices].tolist() + top25[RERANK:]
        all_reranked.append(reranked)
    assert len(all_reranked) == df_test.shape[0]
    df_test["MisconceptionId"] = [" ".join(str(x) for x in row) for row in all_reranked]

    # submit
    df_sub = df_test[["QuestionId_Answer", "MisconceptionId"]]
    df_sub.to_csv("submission.csv", index=False)
    pd.read_csv("submission.csv")  # sanity check


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, type=Path, help="The csv folder"
    )
    parser.add_argument(
        "--top25-path",
        required=True,
        type=str,
        help="Top 25 misconceptions from infer_top25.py (json)",
    )
    parser.add_argument("--model-path", required=True, type=str, help="vLLM model path")
    args = parser.parse_args()
    args = Args(**vars(args))
    print(args)
    t0 = time.perf_counter()
    main(args)
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.2f} secs.")
