from typing import Callable, Union

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict
from prelims import StaticSitePostsHandler
from prelims.processor import Recommender


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    for handler in cfg.handlers:
        h = StaticSitePostsHandler(
            to_absolute_path(handler.target_path), ignore_files=handler.ignore_files
        )
        print(f"target: {to_absolute_path(handler.target_path)}")
        for prc in handler.processors:
            set_processor(h, prc)
        h.execute()


def set_processor(h: StaticSitePostsHandler, cfg: DictConfig) -> StaticSitePostsHandler:
    if cfg.type == "recommender":
        set_recommender(h, cfg)
    else:
        raise NotImplementedError("Unknown Processor")


def set_recommender(
    h: StaticSitePostsHandler, cfg: DictConfig
) -> StaticSitePostsHandler:
    tfidf_opts: dict[
        str, Union[int, str, Callable[[str], list[str]]]
    ] = OmegaConf.to_container(cfg.tfidf_options)
    # Work around for Optional field
    # https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#struct-flag
    with open_dict(cfg):
        cfg.tokenizer = cfg.get("tokenizer", None)
    tokenizer_opt = cfg.tokenizer
    if tokenizer_opt:
        if tokenizer_opt.lang == "ja" and tokenizer_opt.type == "sudachi":
            from .ja.tokenizer import Tokenizer

            tokenizer = Tokenizer(mode=tokenizer_opt.mode, dict=tokenizer_opt.dict)
            tfidf_opts["tokenizer"] = tokenizer.tokenize
        else:
            raise NotImplementedError("Unknown lang/type for tokenizer")

    h.register_processor(Recommender(permalink_base=cfg.permalink_base, **tfidf_opts))
    return h


if __name__ == "__main__":
    main()
