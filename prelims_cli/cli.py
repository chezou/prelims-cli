import logging
import os

import click
from omegaconf import OmegaConf, open_dict
from prelims import StaticSitePostsHandler  # type: ignore

from processor import set_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--config", type=click.Path(exists=True))
def main(config: str) -> None:
    cfg = OmegaConf.load(config)
    for handler in cfg.handlers:
        # Work around for Optional field
        # https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#struct-flag
        with open_dict(handler):
            handler.ignore_files = handler.get("ignore_files", [])
            handler.encoding = handler.get("encoding", "utf-8")
        h = StaticSitePostsHandler(
            handler.target_path,
            ignore_files=handler.ignore_files,
            encoding=handler.encoding,
        )
        logger.info(f"Target Path: {os.path.abspath(handler.target_path)}")
        for prc in handler.processors:
            set_processor(h, prc)
        h.execute()


if __name__ == "__main__":
    main()
