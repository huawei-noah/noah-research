import logging
from typing import Optional

def configure_logger(logger, filename: Optional[str] = None):

    stream_formatter = logging.Formatter(
        "%(asctime)s\t%(levelname)s\t%(name)s\t%(filename)s\t"
        "%(lineno)d\t%(threadName)s\t%(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(stream_formatter)

    logger.setLevel(logging.INFO)
    logger.propagate = False

    if filename is not None:
        file_formatter = logging.Formatter(
            "%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s"
        )
        file_handler = logging.FileHandler(filename, mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.addHandler(stream_handler)