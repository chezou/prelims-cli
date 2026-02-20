from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from prelims_cli import __version__
from prelims_cli.processor import set_processor


def test_version():
    assert isinstance(__version__, str)


def test_set_processor_unknown_type():
    handler = MagicMock()
    cfg = OmegaConf.create({"type": "unknown"})
    try:
        set_processor(handler, cfg)
        assert False, "Expected NotImplementedError"
    except NotImplementedError as e:
        assert "unknown" in str(e).lower()


@patch("prelims_cli.processor.set_recommender")
def test_set_processor_routes_recommender(mock_set_recommender):
    handler = MagicMock()
    cfg = OmegaConf.create({"type": "recommender"})
    set_processor(handler, cfg)
    mock_set_recommender.assert_called_once_with(handler, cfg)


@patch("prelims_cli.processor.set_open_graph_media_extractor")
def test_set_processor_routes_open_graph(mock_set_og):
    handler = MagicMock()
    cfg = OmegaConf.create({"type": "open_graph_media_extractor"})
    set_processor(handler, cfg)
    mock_set_og.assert_called_once_with(handler, cfg)
