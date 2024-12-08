import asyncio
import json
import logging
import os
import random
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import IPython.display as ipd
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm as atqdm

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
semaphore = asyncio.Semaphore(10)


@dataclass
class Args:
    dataset_dir: Path


class Answers(BaseModel):
    a: str
    b: str
    c: str
    d: str


class Misconceptions(BaseModel):
    a: str
    b: str
    c: str
    d: str


class MathQuestion(BaseModel):
    subject: str
    construct_name: str  # can't use `construct` because it shadows pydantic internals
    question: str
    answers: Answers
    misconceptions: Misconceptions


class MathQuestionList(BaseModel):
    math_questions: list[MathQuestion]


def merge_with_misconceptions(
    df_train: pd.DataFrame, df_mis: pd.DataFrame
) -> pd.DataFrame:
    mis_dict = df_mis["MisconceptionName"].to_dict()
    df_merged = df_train.copy()
    for letter in "ABCD":
        df_merged[f"Misconception{letter}Name"] = df_merged[
            f"Misconception{letter}Id"
        ].apply(lambda x: None if np.isnan(x) else mis_dict[int(x)])
    return df_merged


async def gen_synthetic_for_one_q(
    row: pd.Series, client: AsyncOpenAI, max_retries: int = 5
) -> MathQuestionList:
    async with semaphore:
        for retry in range(max_retries):
            try:
                system_prompt = "You are a mathematics teacher tasked to create questions to assess the student's understanding of math concepts. You will be presented with one example: the math question, 1 correct and 3 distraction answers, along with the math misconceptions that led students choosing the distractors instead. Your task is to create similar, but diverse set of 10 new questions. Stick to the given math subject, but feel free to change the construct_name. Remember, each set must contain exactly one '- (answer X is correct, no misconception)'. Return the answer in json."
                prompt = f"""
subject: {row['SubjectName']}

construct_name: {row['ConstructName']}

question: {row['QuestionText']}

answers:
a. {row['AnswerAText']}
b. {row['AnswerBText']}
c. {row['AnswerCText']}
d. {row['AnswerDText']}

misconceptions:
a. {row['MisconceptionAName'] or '- (answer a is correct, no misconception)'}
b. {row['MisconceptionBName'] or '- (answer b is correct, no misconception)'}
c. {row['MisconceptionCName'] or '- (answer c is correct, no misconception)'}
d. {row['MisconceptionDName'] or '- (answer d is correct, no misconception)'}
"""
                completion = await client.beta.chat.completions.parse(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_format=MathQuestionList,
                )
                synthetic_questions = completion.choices[0].message.parsed
                if synthetic_questions is None:
                    raise ValueError("synthetic is None")
            except Exception as e:
                logger.error(f"Error on attempt {retry}: {e}")
                await asyncio.sleep(2**retry)  # exponential backoff
            else:
                return synthetic_questions
        raise RuntimeError(f"Failed after {max_retries} retries.")


async def gen_synthetic(df_merged: pd.DataFrame, savepath: Path):
    client = AsyncOpenAI()
    synthetic_2d: list[MathQuestionList] = await atqdm.gather(
        *[gen_synthetic_for_one_q(row, client) for _, row in df_merged.iterrows()]
    )
    d: list[MathQuestion] = []
    for row in synthetic_2d:
        d += row.math_questions
    with open(savepath, "w") as f:
        json.dump([dd.model_dump() for dd in d], f)


async def main(args: Args):
    # load dataset
    df_train = pd.read_csv(args.dataset_dir / "train.csv")
    df_mis = pd.read_csv(args.dataset_dir / "misconception_mapping.csv")
    df_merged = merge_with_misconceptions(df_train, df_mis)

    # filter only the perfect dataset (a question must be accompanied with 3 misconceptions and 1 correct (nan miscon))
    criteria = (
        df_merged[
            [
                "MisconceptionAId",
                "MisconceptionBId",
                "MisconceptionCId",
                "MisconceptionDId",
            ]
        ]
        .notna()
        .sum(axis=1)
        == 3
    )
    df_merged = df_merged[criteria].reset_index(drop=True).head(4)

    folder = args.dataset_dir / "synthetic"
    folder.mkdir(parents=True, exist_ok=True)
    synthetic_cache = folder / "synthetic.json"

    # generate synthetic!
    if not synthetic_cache.exists():
        logger.info("creating synthetic questions")
        await gen_synthetic(df_merged, synthetic_cache)
    else:
        logger.info("synthetic cache exists, skipping")

    # rebuild everything as id this is the original dataset
    # dont forget to add column


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, type=Path, help="The csv folder."
    )
    args = parser.parse_args()
    args = Args(dataset_dir=args.dataset_dir)
    print(args)
    asyncio.run(main(args))
