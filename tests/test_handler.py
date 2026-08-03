from unittest.mock import MagicMock

from prelims import Post  # type: ignore

from prelims_cli.handler import StablePostsHandler, dump_front_matter, save_post

BLOCK_STYLE = """---
title: hello
tags:
  - a
  - b
---

Hello world.
"""

FLOW_STYLE = """---
title: hello
tags: [a, b]
---

Hello world.
"""

DRAFT = """---
title: hello
draft: true
---

Hello world.
"""

REPEATED_IN_BODY = """---
title: hello
---

Hello world.

title: hello
"""


def write_post(tmp_path, content):
    path = tmp_path / "index.md"
    path.write_text(content, encoding="utf-8")
    return path


def test_dump_front_matter_indents_sequences():
    dumped = dump_front_matter({"title": "hello", "tags": ["a", "b"]})
    assert dumped == "title: hello\ntags:\n  - a\n  - b\n"


def test_dump_front_matter_does_not_wrap_long_values():
    url = "https://example.com/" + "a" * 100
    assert dump_front_matter({"url": url}) == f"url: {url}\n"


def test_save_post_keeps_unchanged_post_untouched(tmp_path):
    path = write_post(tmp_path, BLOCK_STYLE)
    post = Post.load(path)

    post.update("title", "bye", allow_overwrite=False)
    save_post(post)

    assert path.read_text(encoding="utf-8") == BLOCK_STYLE


def test_save_post_keeps_unchanged_flow_style_post_untouched(tmp_path):
    path = write_post(tmp_path, FLOW_STYLE)
    post = Post.load(path)

    save_post(post)

    assert path.read_text(encoding="utf-8") == FLOW_STYLE


def test_save_post_writes_block_style(tmp_path):
    path = write_post(tmp_path, FLOW_STYLE)
    post = Post.load(path)

    post.update("recommendations", ["/post/x/", "/post/y/"])
    save_post(post)

    assert (
        path.read_text(encoding="utf-8")
        == """---
title: hello
tags:
  - a
  - b
recommendations:
  - /post/x/
  - /post/y/
---

Hello world.
"""
    )


def test_save_post_only_replaces_the_front_matter(tmp_path):
    path = write_post(tmp_path, REPEATED_IN_BODY)
    post = Post.load(path)

    post.update("recommendations", ["/post/x/"])
    save_post(post)

    assert (
        path.read_text(encoding="utf-8")
        == """---
title: hello
recommendations:
  - /post/x/
---

Hello world.

title: hello
"""
    )


def test_save_post_skips_draft(tmp_path):
    path = write_post(tmp_path, DRAFT)
    post = Post.load(path)

    post.update("recommendations", ["/post/x/"])
    save_post(post)

    assert path.read_text(encoding="utf-8") == DRAFT


def test_save_posts_saves_every_post(tmp_path):
    paths = []
    for i in range(2):
        path = tmp_path / f"{i}.md"
        path.write_text(FLOW_STYLE, encoding="utf-8")
        paths.append(path)

    posts = [Post.load(path) for path in paths]
    for post in posts:
        post.update("recommendations", ["/post/x/"])
    StablePostsHandler.save_posts(posts)

    for path in paths:
        assert "recommendations:\n  - /post/x/\n" in path.read_text(encoding="utf-8")


def test_save_posts_is_used_by_execute(tmp_path):
    write_post(tmp_path, FLOW_STYLE)
    h = StablePostsHandler(str(tmp_path))
    h.save_posts = MagicMock()

    h.execute()

    h.save_posts.assert_called_once()
