# prelims-cli

CLI for [prelims](https://github.com/takuti/prelims).

## Install

Run:

```sh
pip install prelims-cli
```

If you need Japanese tokenization, run:

```sh
pip install prelims-cli[ja]
```

## Usage

Assuming the following folder directory:

```sh
- content
|  ├── post
|  └── blog
└─ scripts
   └ config
     └ myconfig.yaml
```

where, post and blog are pages, and scripts is the place to put scripts.

Here is the example of configuration:

```myconfig.yaml
handlers:
  - target_path: "content/blog"
    ignore_files:
      - _index.md
    processors:
      - type: recommender
        permalink_base: "/blog"
        tfidf_options:
          stop_words: english
          max_df: 0.95
          min_df: 2
        tokenizer: null
  - target_path: "content/post"
    ignore_files:
      - _index.md
    processors:
      - type: recommender
        permalink_base: "/post"
        tfidf_options:
          max_df: 0.95
          min_df: 2
        tokenizer:
          lang: ja
          type: sudachi
          mode: C
          dict: full
```

```sh
$ prelims-cli --config ./scripts/config/myconfig.yaml
target: /user/chezo/src/chezo.uno/content/blog
target: /users/chezo/src/chezo.uno/content/post
```

Then your articles' front matter were updated.
