import pytest

from src.index_builder import (
    _clean_page_markers,
    _contextualize_text,
    _extract_chunk_pages,
    preprocess_for_bm25,
)


pytestmark = pytest.mark.unit


class TestCleanPageMarkers:
    def test_removes_single_marker(self):
        assert _clean_page_markers("hello --- Page 42 --- world") == "hello  world"

    def test_removes_multiple_markers(self):
        text = "start --- Page 1 --- middle --- Page 2 --- end"
        assert "--- Page" not in _clean_page_markers(text)

    def test_strips_whitespace(self):
        assert _clean_page_markers("  text  ") == "text"

    def test_empty_string(self):
        assert _clean_page_markers("") == ""


class TestContextualizeText:
    def test_single_page(self):
        result = _contextualize_text("Chapter 1 Intro", [5], "Some content")
        assert "Section Path: Chapter 1 Intro" in result
        assert "Page Span: page 5" in result
        assert "Content:\nSome content" in result

    def test_multi_page(self):
        result = _contextualize_text("Chapter 2 Trees", [10, 12], "B+ tree content")
        assert "Page Span: pages 10-12" in result

    def test_empty_pages(self):
        result = _contextualize_text("Chapter 3", [], "No pages")
        assert "Page Span: unknown pages" in result

    def test_deduplicates_and_sorts_pages(self):
        result = _contextualize_text("Ch1", [3, 1, 3, 2], "text")
        assert "pages 1-3" in result


class TestExtractChunkPages:
    def test_no_markers_uses_current_page(self):
        pages, updated = _extract_chunk_pages("plain text", 5)
        assert pages == [5]
        assert updated == 5

    def test_single_marker_advances_page(self):
        pages, updated = _extract_chunk_pages("before --- Page 10 --- after", 1)
        assert 11 in pages
        assert updated == 11

    def test_multiple_markers(self):
        text = "start --- Page 3 --- middle --- Page 5 --- end"
        pages, updated = _extract_chunk_pages(text, 1)
        assert 4 in pages
        assert 6 in pages
        assert updated == 6

    def test_empty_content_after_marker_excluded(self):
        pages, updated = _extract_chunk_pages("text --- Page 7 --- ", 1)
        assert 1 in pages
        assert updated == 8


class TestPreprocessForBm25:
    def test_lowercases(self):
        tokens = preprocess_for_bm25("Hello World")
        assert tokens == ["hello", "world"]

    def test_strips_punctuation(self):
        tokens = preprocess_for_bm25("B+ tree's hash#tag")
        assert "b+" in tokens
        assert "tree's" in tokens
        assert "hash#tag" in tokens

    def test_empty_string(self):
        assert preprocess_for_bm25("") == []
