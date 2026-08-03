import re
from typing import Any

import yaml
from prelims import StaticSitePostsHandler  # type: ignore

RE_FRONT_MATTER = re.compile(r"---\n([\s\S]*?\n)---\n")


class BlockStyleDumper(yaml.Dumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False) -> Any:
        # PyYAML emits `tags:\n- a`, while editors and CMSes write `tags:\n  - a`
        return super().increase_indent(flow, False)


def dump_front_matter(front_matter: Any) -> str:
    return yaml.dump(
        front_matter,
        Dumper=BlockStyleDumper,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=float("inf"),
    )


def save_post(post: Any) -> None:
    if not post.is_valid():
        return

    m = RE_FRONT_MATTER.search(post.raw_content)
    if m is None:
        return

    # Rewriting an untouched post only reformats it
    if post.front_matter == yaml.safe_load(m.group(1)):
        return

    content = post.raw_content.replace(m.group(1), dump_front_matter(post.front_matter))
    with open(post.path, "w", encoding=post.encoding) as f:
        f.write(content)


class StablePostsHandler(StaticSitePostsHandler):  # type: ignore
    """Handler that keeps the front matter formatting stable across runs.

    prelims re-dumps the front matter of every post it loads, so a run over
    unchanged articles still rewrites them in a different YAML style. This
    handler writes a post only when a processor updated its front matter, and
    dumps it in the block style editors and CMSes produce.
    """

    @staticmethod
    def save_posts(posts: list[Any]) -> None:
        for post in posts:
            save_post(post)
