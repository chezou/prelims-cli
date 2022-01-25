from typing import Callable, Union, cast
import logging
import os

import click
from omegaconf import DictConfig, OmegaConf, open_dict
from prelims import StaticSitePostsHandler  # type: ignore
from prelims.processor import Recommender  # type: ignore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prelims-cli")


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


def set_processor(h: StaticSitePostsHandler, cfg: DictConfig) -> None:
    if cfg.type == "recommender":
        set_recommender(h, cfg)
    else:
        raise NotImplementedError(f"Unknown Processor type: {cfg.type}")


TokenizerCallableType = Callable[[str], list[str]]
TfIdfValueType = Union[int, float, str, TokenizerCallableType]


def set_recommender(h: StaticSitePostsHandler, cfg: DictConfig) -> None:
    tfidf_opts = cast(
        dict[str, TfIdfValueType],
        OmegaConf.to_container(cfg.tfidf_options),
    )
    # Work around for Optional field
    # https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#struct-flag
    with open_dict(cfg):
        cfg.tokenizer = cfg.get("tokenizer", None)
        if not cfg.tokenizer:
            logger.warning("tokenizer is undefined")
        cfg.topk = cfg.get("topk", 3)
        if not cfg.topk:
            logger.warning("topk is undefined. Fallback to default: 3")
        cfg.lower_path = cfg.get("lower_path", True)
        if not cfg.lower_path:
            logger.warning("lower_path is undefined. Fallback to default: True")
    tkn_opt = cfg.tokenizer
    if tkn_opt:
        if tkn_opt.lang == "ja" and tkn_opt.type == "sudachi":
            from .ja.tokenizer import Tokenizer

            tokenizer = Tokenizer(mode=tkn_opt.mode, dict=tkn_opt.dict)
            tfidf_opts["tokenizer"] = tokenizer.tokenize
        else:
            raise NotImplementedError(
                f"Unknown lang {tkn_opt.lang} or type {tkn_opt.type} for tokenizer"
            )

    h.register_processor(
        Recommender(
            permalink_base=cfg.permalink_base,
            topk=cfg.topk,
            lower_path=cfg.lower_path,
            **tfidf_opts,
        )
    )


if __name__ == "__main__":
    main()
