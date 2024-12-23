import logging
import sys
from pathlib import Path

from .utils import get_env_variable

debug_mode = 0


def setup_logging(log_file: str = "app.log"):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    level = logging.INFO if debug_mode == "0" else logging.DEBUG
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.FileHandler(log_dir / log_file), logging.StreamHandler(sys.stdout)],
    )

    return logging.getLogger("rag_chat")
