import os
from datetime import datetime

import pytz


def wib_now() -> str:
    """Generate current WIB time. Quite easy to read imho.
    Example: 2024-11-23__17.30.02
    """
    wib = pytz.timezone("Asia/Jakarta")
    timestamp = datetime.now(wib).strftime("%Y-%m-%d__%H.%M.%S")
    return timestamp


def local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def is_rank_0() -> bool:
    return local_rank() == 0
