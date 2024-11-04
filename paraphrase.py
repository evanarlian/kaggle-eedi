import asyncio
import json
import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

# load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class Args:
    dataset_dir: Path


semaphore = asyncio.Semaphore(10)


class Paraphrases(BaseModel):
    paraphrases: list[str]


async def paraphrase_single(
    text: str, client: AsyncOpenAI, system_prompt: str, max_retries: int = 5
) -> Paraphrases:
    # import random

    # if random.random() < 0.02:
    #     raise RuntimeError("fake error")
    # return Paraphrases(paraphrases=["meme", "lol"])
    async with semaphore:
        for retry in range(max_retries):
            try:
                completion = await client.beta.chat.completions.parse(
                    # model="gpt-4o-2024-08-06",
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                    ],
                    response_format=Paraphrases,
                )
                paraphrased = completion.choices[0].message.parsed
                if paraphrased is None:
                    raise ValueError("paraphrased is None")
            except Exception as e:
                logger.error(f"Error on attempt {retry}: {e}")
                await asyncio.sleep(2**retry)  # exponential backoff
            else:
                return paraphrased
        raise RuntimeError(f"Failed after {max_retries} retries.")


async def paraphrase(texts: list[str], system_prompt: str, savepath: Path):
    # client = AsyncOpenAI()
    # TODO with semaphore and exponential backoff, there is no need to save to disk first
    with open(savepath, "r") as f:
        # {0: [], 1: [], ...}
        content = json.load(f)
    missing_indices = [k for k, v in content.items() if v == []]
    tasks = []
    for missing in missing_indices:
        task = paraphrase_single(texts[missing], client, system_prompt)
        tasks.append(task)
    result = asyncio.gather(
        tasks,
    )
    pass


async def main(args: Args):
    # load dataset
    df_train = pd.read_csv(args.dataset_dir / "train.csv")
    df_train = df_train.sort_values("QuestionId")
    questions = df_train["QuestionText"].tolist()

    df_mis = pd.read_csv(args.dataset_dir / "misconception_mapping.csv")
    df_mis = df_mis.sort_values("MisconceptionId")
    misconceptions = df_mis["MisconceptionName"].tolist()

    # premade the json
    # TODO with semaphore and exponential backoff, there is no need to premade
    folder = args.dataset_dir / "paraphrased"
    folder.mkdir(parents=True, exist_ok=True)
    question_path = folder / "question.json"
    if not question_path.exists():
        with open(question_path, "w") as f:
            empty = {i: [] for i in range(len(questions))}
            json.dump(empty, f)
    misconception_path = folder / "misconception.json"
    if not misconception_path.exists():
        with open(misconception_path, "w") as f:
            empty = {i: [] for i in range(len(misconceptions))}
            json.dump(empty, f)

    # paraphrase but skip already made paraphrases
    Q_PROMPT = "Paraphrase the question below without changing key information. Do not answer the question. Make 4 (four) paraphrases."
    M_PROMPT = "Paraphrase the math misconception below without changing key information. Make 4 (four) paraphrases."

    await paraphrase(questions, Q_PROMPT, question_path)
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, type=Path, help="The csv folder."
    )
    args = parser.parse_args()
    args = Args(dataset_dir=args.dataset_dir)
    print(args)
    asyncio.run(main(args))
