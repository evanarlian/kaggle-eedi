import asyncio
import json
import logging
import os
import random
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import IPython.display as ipd
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm as atqdm

from eedi.datasets import make_complete_query, make_nice_df

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
semaphore = asyncio.Semaphore(10)


@dataclass
class Args:
    dataset_dir: Path


class Generated(BaseModel):
    question: str
    correct_answer: str
    wrong_answer: str


class GeneratedList(BaseModel):
    generated_list: list[Generated]


class Datapoint(BaseModel):
    subject_name: str
    construct_name: str
    question: str
    correct_answer: str
    wrong_answer: str
    misconception_name: str


async def gen_synthetic_for_miscon(
    row: pd.Series, mis_text: str, client: AsyncOpenAI, max_retries: int = 5
) -> list[Datapoint]:
    async with semaphore:
        for retry in range(max_retries):
            try:
                system_prompt = "You are a mathematics teacher tasked to create questions to assess the student's understanding of math concepts. You will be presented with one example: the math question, the correct and wrong answer, along with the math misconceptions that led students choosing wrong answer instead of the correct one. Your task is to create similar, but diverse set of new questions, correct and wrong answers according to the given subject, construct, and misconception. Make sure the generated contents are diverse enough. Generate 5 set. Output answer in json."
                prompt = f"""# One-shot example:
subject: {row['SubjectName']}

construct: {row['ConstructName']}

question: {row['QuestionText']}

correct_answer: {row['CorrectText']}

wrong_answer: {row['WrongText']}

misconception: {mis_text}


# Create new questions based on given example
subject: {row['SubjectName']}

construct: {row['ConstructName']}

question: ...

correct_answer: ...

wrong_answer: ...

misconception: {mis_text}
"""
                completion = await client.beta.chat.completions.parse(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_format=GeneratedList,
                )
                generated_list = completion.choices[0].message.parsed
                if generated_list is None:
                    raise ValueError("generated_list is None")
            except Exception as e:
                logger.error(f"Error on attempt {retry}: {e}")
                await asyncio.sleep(2**retry)  # exponential backoff
            else:
                return [
                    Datapoint(
                        subject_name=row["SubjectName"],
                        construct_name=row["ConstructName"],
                        question=gen.question,
                        correct_answer=gen.correct_answer,
                        wrong_answer=gen.wrong_answer,
                        misconception_name=mis_text,
                    )
                    for gen in generated_list.generated_list
                ]
        raise RuntimeError(f"Failed after {max_retries} retries.")


async def gen_synthetic(
    df_train: pd.DataFrame, orig_mis: list[str], n_generate: int, savepath: Path
):
    # using inverse sampling from train in order to balance which misconception to pick
    mis_id_to_train_idx = defaultdict(list)
    for train_idx, mis_id in enumerate(df_train["MisconceptionId"]):
        mis_id_to_train_idx[mis_id].append(train_idx)
    mis_id_to_count = {k: len(v) for k, v in mis_id_to_train_idx.items()}
    count_arr = np.array(list(mis_id_to_count.values()))
    inv_prob = 1 / count_arr
    inv_prob = inv_prob / inv_prob.sum()
    to_generate = np.random.choice(
        np.array(list(mis_id_to_count.keys())),
        size=n_generate,
        replace=True,
        p=inv_prob,
    )
    # generate
    client = AsyncOpenAI()
    tasks = []
    for mis_id in to_generate:
        row_idx = random.choice(mis_id_to_train_idx[mis_id])
        row = df_train.loc[row_idx]
        mis_text = orig_mis[row["MisconceptionId"]]
        tasks.append(gen_synthetic_for_miscon(row, mis_text, client))

    datapoint_2d: list[list[Datapoint]] = await atqdm.gather(*tasks)
    d = [
        datapoint.model_dump()
        for datapoints in datapoint_2d
        for datapoint in datapoints
    ]
    with open(savepath, "w") as f:
        json.dump(d, f)


async def main(args: Args):
    # load dataset
    df_train = pd.read_csv(args.dataset_dir / "train.csv")
    df_train = make_nice_df(df_train)
    df_mis = pd.read_csv(args.dataset_dir / "misconception_mapping.csv")
    orig_mis = df_mis["MisconceptionName"].tolist()

    folder = args.dataset_dir / "synthetic"
    folder.mkdir(parents=True, exist_ok=True)
    synthetic_cache = folder / "synthetic.json"

    # generate synthetic data from known example in train
    if not synthetic_cache.exists():
        logger.info("generating questions")
        await gen_synthetic(
            df_train, orig_mis, n_generate=len(df_train), savepath=synthetic_cache
        )
    else:
        logger.info("generation cache exists, skipping")

    # generate syntheticc data from misconception ids NOT in the train dataset
    # TODO
    # rebuild everything as if this is the original dataset
    # dont forget to add column (ai created or whatever)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, type=Path, help="The csv folder."
    )
    args = parser.parse_args()
    args = Args(dataset_dir=args.dataset_dir)
    print(args)
    asyncio.run(main(args))
