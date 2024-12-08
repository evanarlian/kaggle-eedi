import asyncio
import json
import logging
import random
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm as atqdm

from eedi.datasets import make_nice_df

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
semaphore = asyncio.Semaphore(10)


@dataclass
class Args:
    dataset_dir: Path


class GeneratedKnown(BaseModel):
    question: str
    correct_answer: str
    wrong_answer: str


class GeneratedKnownList(BaseModel):
    generated_list: list[GeneratedKnown]


class GeneratedUnknown(BaseModel):
    subject_name: str
    construct_name: str
    question: str
    correct_answer: str
    wrong_answer: str


class GeneratedUnknownList(BaseModel):
    generated_list: list[GeneratedUnknown]


class Datapoint(BaseModel):
    subject_name: str
    construct_name: str
    question: str
    correct_answer: str
    wrong_answer: str
    misconception_name: str


async def gen_known_synthetic_impl(
    row: pd.Series, mis_text: str, client: AsyncOpenAI
) -> list[Datapoint]:
    async with semaphore:
        for retry in range(1000):
            try:
                system_prompt = "You are a mathematics teacher tasked to create questions to assess the student's understanding of math concepts. You will be presented with one example: the math question, the correct and wrong answer, along with the math misconceptions that led students choosing wrong answer instead of the correct one. Your task is to create similar, but diverse set of new questions, correct and wrong answers according to the given subject, construct, and misconception. Make sure the generated contents are diverse enough. Generate 5 sets. Output answer in json."
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
                    temperature=0.7,
                    top_p=0.7,
                    response_format=GeneratedKnownList,
                )
                generated_list = completion.choices[0].message.parsed
                if generated_list is None:
                    raise ValueError("generated_list is None")
            except Exception as e:
                logger.error(f"Error on attempt {retry}: {e}")
                await asyncio.sleep(5)  # flat delay
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
        raise RuntimeError("Failed after 1000 retries.")


async def gen_known_synthetic(
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
        tasks.append(gen_known_synthetic_impl(row, mis_text, client))

    datapoint_2d: list[list[Datapoint]] = await atqdm.gather(*tasks)
    d = [
        datapoint.model_dump()
        for datapoints in datapoint_2d
        for datapoint in datapoints
    ]
    with open(savepath, "w") as f:
        json.dump(d, f)


async def gen_unknown_synthetic_impl(
    mis_text: str, client: AsyncOpenAI
) -> list[Datapoint]:
    async with semaphore:
        for retry in range(1000):
            try:
                system_prompt = "You are a mathematics teacher tasked to create questions to assess the student's understanding of math concepts."
                prompt = f"""# Few-shot example 1
subject: Completing the Square

construct: Complete the square for expressions that end up in the form (x + a)² + b

question: When Sarah completes the square, what should replace the triangle?
\\[
p^{2}-10 p-1 \\equiv(p-5)^{2}  \\Delta
\\]

correct_answer: \\( \\Delta=-26 \\)

wrong_answer: \\( \\Delta=+24 \\)

misconception: When completing the square in the form (x - a)^2 + b, believes b = a^2 + the original constant


# Few-shot example 2
subject: Square Roots, Cube Roots, etc

construct: Recognise square roots

question: Which of the following is the square root of \\( 81 \\) ?

correct_answer: \\( 9 \\)

wrong_answer: \\( 3 \\)

misconception: Mixes up square rooting and fourth rooting


# Your task
subject_name: ...

construct_name: ...

question: ...

correct_answer: ...

wrong_answer: ...

misconception: {mis_text}


For context, these questions are designed to test student math understanding. They might have some misconceptions, which led them to select incorrect answer.
Similar to 2 few-shot examples above, given misconception, your task is to generate these 5 things: subject_name, construct_name, question, correct_answer, and wrong_answer.
Feel free to choose the math subject_name and construct_name suitable for high school students. Pay attention to the misconception, your generations should be relatable to the said misconception.
Make sure the generated contents are diverse enough. Generate 5 sets. Output answer in json.
"""
                completion = await client.beta.chat.completions.parse(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_format=GeneratedUnknownList,
                )
                generated_list = completion.choices[0].message.parsed
                if generated_list is None:
                    raise ValueError("generated_list is None")
            except Exception as e:
                logger.error(f"Error on attempt {retry}: {e}")
                await asyncio.sleep(5)  # flat delay
            else:
                return [
                    Datapoint(
                        subject_name=gen.subject_name,
                        construct_name=gen.construct_name,
                        question=gen.question,
                        correct_answer=gen.correct_answer,
                        wrong_answer=gen.wrong_answer,
                        misconception_name=mis_text,
                    )
                    for gen in generated_list.generated_list
                ]
        raise RuntimeError("Failed after 1000 retries.")


async def gen_unknown_synthetic(
    df_train: pd.DataFrame, orig_mis: list[str], savepath: Path
):
    # find the misconceptions not present in train dataset
    mis_in_train = set(df_train["MisconceptionId"])
    all_mis = set(list(range(len(orig_mis))))
    not_in_train = all_mis - mis_in_train
    # generate
    client = AsyncOpenAI()
    tasks = []
    for mis_id in not_in_train:
        mis_text = orig_mis[mis_id]
        # it has been calculated that we need to call this 2.7 times in order to roughly match the known case
        # 4370 samples in train / 1604 misconceptions in train ~ 2.7 samples per misconception
        # i just choose 2 because it is cheaper to run, we can't reliably predict the unknown case anyway
        tasks.append(gen_unknown_synthetic_impl(mis_text, client))
        tasks.append(gen_unknown_synthetic_impl(mis_text, client))
    datapoint_2d: list[list[Datapoint]] = await atqdm.gather(*tasks)
    d = [
        datapoint.model_dump()
        for datapoints in datapoint_2d
        for datapoint in datapoints
    ]
    with open(savepath, "w") as f:
        json.dump(d, f)


def rebuild(
    df_train: pd.DataFrame,
    orig_mis: list[str],
    synthetic_known_cache: Path,
    synthetic_unknown_cache: Path,
    savepath: Path,
) -> None:
    with open(synthetic_known_cache, "r") as f:
        synthetic_known = json.load(f)
    with open(synthetic_unknown_cache, "r") as f:
        synthetic_unknown = json.load(f)

    subject_to_subject_id = {}
    for i, s_name, s_id in df_train[["SubjectName", "SubjectId"]].itertuples():
        subject_to_subject_id[s_name] = s_id
    construct_to_construct_id = {}
    for i, c_name, c_id in df_train[["ConstructName", "ConstructId"]].itertuples():
        construct_to_construct_id[c_name] = c_id
    question_to_question_id = {}
    for i, q_text, q_id in df_train[["QuestionText", "QuestionId"]].itertuples():
        question_to_question_id[q_text] = q_id
    mis_to_mis_id = {mis_text: mis_id for mis_id, mis_text in enumerate(orig_mis)}

    new_train_rows = []
    for known in synthetic_known:
        new_q = known["question"]
        # NOTE: this is bad slow code but i dont care
        if new_q not in question_to_question_id:
            question_to_question_id[new_q] = max(question_to_question_id.values()) + 1
        new_train_rows.append(
            {
                "QuestionId": question_to_question_id[new_q],
                "ConstructId": construct_to_construct_id[known["construct_name"]],
                "ConstructName": known["construct_name"],
                "SubjectId": subject_to_subject_id[known["subject_name"]],
                "SubjectName": known["subject_name"],
                "CorrectChoice": "?",
                "CorrectText": known["correct_answer"],
                "QuestionText": new_q,
                "WrongChoice": "?",
                "WrongText": known["wrong_answer"],
                "MisconceptionId": mis_to_mis_id[known["misconception_name"]],
                "QuestionId_Answer": "?",
                "Synthetic": True,
            }
        )
    for unknown in synthetic_unknown:
        # NOTE: these are bad slow code but i dont care
        new_subj = unknown["subject_name"]
        new_c = unknown["construct_name"]
        new_q = unknown["question"]
        if new_subj not in subject_to_subject_id:
            subject_to_subject_id[new_subj] = max(subject_to_subject_id.values()) + 1
        if new_c not in construct_to_construct_id:
            construct_to_construct_id[new_c] = (
                max(construct_to_construct_id.values()) + 1
            )
        if new_q not in question_to_question_id:
            question_to_question_id[new_q] = max(question_to_question_id.values()) + 1
        new_train_rows.append(
            {
                "QuestionId": question_to_question_id[new_q],
                "ConstructId": construct_to_construct_id[new_c],
                "ConstructName": new_c,
                "SubjectId": subject_to_subject_id[new_subj],
                "SubjectName": new_subj,
                "CorrectChoice": "?",
                "CorrectText": unknown["correct_answer"],
                "QuestionText": new_q,
                "WrongChoice": "?",
                "WrongText": unknown["wrong_answer"],
                "MisconceptionId": mis_to_mis_id[unknown["misconception_name"]],
                "QuestionId_Answer": "?",
                "Synthetic": True,
            }
        )

    df_train_new = pd.DataFrame.from_records(new_train_rows)
    df_train_all = pd.concat([df_train, df_train_new]).reset_index(drop=True)
    df_train_all.to_csv(savepath, index=False)
    logger.info(f"saved usable synthetic dataframe in {savepath}")


async def main(args: Args):
    # load dataset
    df_train = pd.read_csv(args.dataset_dir / "train.csv")
    df_train = make_nice_df(df_train)
    df_mis = pd.read_csv(args.dataset_dir / "misconception_mapping.csv")
    orig_mis = df_mis["MisconceptionName"].tolist()

    folder = args.dataset_dir / "synthetic"
    folder.mkdir(parents=True, exist_ok=True)
    synthetic_known_cache = folder / "synthetic_known.json"
    synthetic_unknown_cache = folder / "synthetic_unknown.json"

    # generate synthetic data from known example in train
    if not synthetic_known_cache.exists():
        logger.info("generating known questions")
        await gen_known_synthetic(
            df_train, orig_mis, n_generate=len(df_train), savepath=synthetic_known_cache
        )
    else:
        logger.info("cache of known generation exists, skipping")

    # generate synthetic data with misconception not included in train
    if not synthetic_unknown_cache.exists():
        logger.info("generating unknown questions")
        await gen_unknown_synthetic(
            df_train, orig_mis, savepath=synthetic_unknown_cache
        )
    else:
        logger.info("cache of unknown generation exists, skipping")

    # rebuild everything as if this is the original dataset, also add synthetic indicator
    rebuild(
        df_train,
        orig_mis,
        synthetic_known_cache,
        synthetic_unknown_cache,
        savepath=folder / "train.csv",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, type=Path, help="The csv folder."
    )
    args = parser.parse_args()
    args = Args(dataset_dir=args.dataset_dir)
    print(args)
    asyncio.run(main(args))
