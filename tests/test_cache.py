"""On-disk response cache (SQLite in a temp folder)."""

from pathlib import Path

from research_agent.cache import ResponseCache


def test_set_then_get(tmp_path: Path):
    cache = ResponseCache(tmp_path / "c.sqlite", ttl_seconds=60)
    cache.set("k", {"a": 1})
    assert cache.get("k") == {"a": 1}
    assert (cache.hits, cache.misses) == (1, 0)


def test_expired_entries_are_misses(tmp_path: Path):
    cache = ResponseCache(tmp_path / "c.sqlite", ttl_seconds=-1)  # everything already expired
    cache.set("k", {"a": 1})
    assert cache.get("k") is None
    assert cache.misses == 1


def test_key_ignores_param_order_and_auth():
    a = ResponseCache.make_key("/works", {"search": "x", "per_page": 5, "api_key": "one"})
    b = ResponseCache.make_key("/works", {"per_page": 5, "search": "x", "api_key": "two"})
    c = ResponseCache.make_key("/works", {"search": "y", "per_page": 5})
    assert a == b  # same question, different caller -> same entry
    assert a != c


def test_persists_across_instances(tmp_path: Path):
    path = tmp_path / "c.sqlite"
    first = ResponseCache(path, ttl_seconds=60)
    first.set("k", {"a": 1})
    first.close()
    assert ResponseCache(path, ttl_seconds=60).get("k") == {"a": 1}
