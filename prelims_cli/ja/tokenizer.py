from typing import Optional

from sudachipy import dictionary, tokenizer  # type: ignore


class Tokenizer:
    def __init__(
        self,
        mode: str = "C",
        dict: str = "core",
        max_df: int = 15,
        min_df: int = 2,
        filter_pos: Optional[list] = None,
    ):
        self.tokenizer_obj = dictionary.Dictionary(dict=dict).create()
        self.max_df = max_df
        self.min_df = min_df
        if filter_pos is None:
            filter_pos = ["名詞", "形容詞"]
        self.filter_pos = filter_pos
        if mode == "A":
            self.mode = tokenizer.Tokenizer.SplitMode.A
        elif mode == "B":
            self.mode = tokenizer.Tokenizer.SplitMode.B
        else:
            self.mode = tokenizer.Tokenizer.SplitMode.C

    def tokenize(self, text: str) -> list[str]:
        return [
            m.surface()
            for m in self.tokenizer_obj.tokenize(text, self.mode)
            if m.part_of_speech()[0] in self.filter_pos
            and self.min_df <= len(m.surface()) <= self.max_df
        ]
