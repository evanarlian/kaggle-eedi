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
from tqdm.asyncio import tqdm as atqdm

from utils import make_nice_df

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
semaphore = asyncio.Semaphore(10)


@dataclass
class Args:
    dataset_dir: Path


class Paraphrases(BaseModel):
    paraphrases: list[str]


async def paraphrase_single(
    text: str, client: AsyncOpenAI, system_prompt: str, max_retries: int = 5
) -> Paraphrases:
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
    client = AsyncOpenAI()
    paraphrased_2d = await atqdm.gather(
        *[paraphrase_single(text, client, system_prompt) for text in texts]
    )
    d = {i: p.paraphrases for i, p in enumerate(paraphrased_2d)}
    with open(savepath, "w") as f:
        json.dump(d, f)


async def main(args: Args):
    # load dataset
    df_train = pd.read_csv(args.dataset_dir / "train.csv")
    df_train = df_train.sort_values("QuestionId")
    questions = df_train["QuestionText"].tolist()

    df_mis = pd.read_csv(args.dataset_dir / "misconception_mapping.csv")
    df_mis = df_mis.sort_values("MisconceptionId")
    misconceptions = df_mis["MisconceptionName"].tolist()

    folder = args.dataset_dir / "paraphrased"
    folder.mkdir(parents=True, exist_ok=True)
    question_path = folder / "question.json"
    misconception_path = folder / "misconception.json"

    Q_PROMPT = "Paraphrase the question below without changing key information. Do not answer the question. Make 4 (four) paraphrases."
    M_PROMPT = "Paraphrase the math misconception below without changing key information. Make 4 (four) paraphrases."

    # paraphrase
    if not question_path.exists():
        logger.info("paraphrasing questions")
        await paraphrase(questions, Q_PROMPT, question_path)
    else:
        logger.info("paraphrased questions exist, skipping")
    if not misconception_path.exists():
        logger.info("paraphrasing misconceptions")
        await paraphrase(misconceptions, M_PROMPT, misconception_path)
    else:
        logger.info("paraphrased misconceptions exist, skipping")

    # mix paraphrased with train dataset
    logger.info("mixing paraphrased with train dataset")
    with open(args.dataset_dir / "paraphrased/question.json", "r") as f:
        Q = json.load(f)
    with open(args.dataset_dir / "paraphrased/misconception.json", "r") as f:
        M = json.load(f)
    df_train = make_nice_df(df_train)
    df_train = pd.merge(df_train, df_mis, on="MisconceptionId")
    df_train = df_train.rename(columns={"MisconceptionName": "MisconceptionText"})

    def inject_para_questions(x, Q):
        q_list = [x["QuestionText"]] + Q[str(x["QuestionId"])]
        ai_created = [False] + [True] * (len(q_list) - 1)
        x["QuestionText"] = q_list
        x["QuestionAiCreated"] = ai_created
        return x

    def inject_para_misconceptions(x, M):
        m_list = [x["MisconceptionText"]] + M[str(x["MisconceptionId"])]
        ai_created = [False] + [True] * (len(m_list) - 1)
        x["MisconceptionText"] = m_list
        x["MisconceptionAiCreated"] = ai_created
        return x

    df_train = df_train.apply(lambda x: inject_para_questions(x, Q), axis=1)
    df_train = df_train.explode(["QuestionText", "QuestionAiCreated"])  # type: ignore
    df_train = df_train.apply(lambda x: inject_para_misconceptions(x, M), axis=1)
    df_train = df_train.explode(["MisconceptionText", "MisconceptionAiCreated"])

    # save
    logger.info(f"saving df_train ({df_train.shape})")
    df_train.to_csv(args.dataset_dir / "paraphrased/train.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, type=Path, help="The csv folder."
    )
    args = parser.parse_args()
    args = Args(dataset_dir=args.dataset_dir)
    print(args)
    asyncio.run(main(args))
