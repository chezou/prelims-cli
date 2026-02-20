import logging
from typing import Callable, Union, cast

from omegaconf import DictConfig, OmegaConf, open_dict
from prelims import StaticSitePostsHandler  # type: ignore
from prelims.processor import OpenGraphMediaExtractor, Recommender  # type: ignore

logger = logging.getLogger(__name__)


def set_processor(h: StaticSitePostsHandler, cfg: DictConfig) -> None:
    if cfg.type == "recommender":
        set_recommender(h, cfg)
    elif cfg.type == "open_graph_media_extractor":
        set_open_graph_media_extractor(h, cfg)
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


def set_open_graph_media_extractor(h: StaticSitePostsHandler, cfg: DictConfig) -> None:
    with open_dict(cfg):
        cfg.image_base = cfg.get("image_base", None)
        cfg.audio_base = cfg.get("audio_base", None)
        cfg.video_base = cfg.get("video_base", None)

    h.register_processor(
        OpenGraphMediaExtractor(
            image_base=cfg.image_base,
            audio_base=cfg.audio_base,
            video_base=cfg.video_base,
        )
    )
