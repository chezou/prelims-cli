from typing import Any

from omegaconf import DictConfig, OmegaConf
from prelims import StaticSitePostsHandler
from prelims.processor import Recommender

import hydra
from hydra.utils import to_absolute_path


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    for handler in cfg.handlers:
        h = StaticSitePostsHandler(
            to_absolute_path(handler.target_path), ignore_files=handler.ignore_files
        )
        print(f"target: {to_absolute_path(handler.target_path)}")
        for prc in handler.processors:
            tfidf_opts: dict[str, Any] = OmegaConf.to_container(prc.tfidf_options)
            tokenizer_opt = prc.tokenizer
            if tokenizer_opt:
                if tokenizer_opt.lang == "ja" and tokenizer_opt.type == "sudachi":
                    from ja.tokenizer import Tokenizer

                    tokenizer = Tokenizer(
                        mode=tokenizer_opt.mode, dict=tokenizer_opt.dict
                    )
                    tfidf_opts["tokenizer"] = tokenizer.tokenize
                else:
                    raise NotImplementedError("Unknown lang/type for tokenizer")

            h.register_processor(
                Recommender(permalink_base=prc.permalink_base, **tfidf_opts)
            )
        h.execute()


if __name__ == "__main__":
    main()
