"""The content-addressed result cache."""

from __future__ import annotations

import json

import pytest

from backend.pipeline import cache
from backend.pipeline.payloads import DEFAULT_RUN, ProtocolSpec, RunSpec, run_hash


@pytest.fixture(autouse=True)
def temp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    yield


def test_miss_returns_none():
    assert cache.load("identify", "deadbeef") is None


def test_round_trip_is_byte_identical():
    payload = {"mare": 9.34, "rows": [{"parameter": "EA", "error_pct": 1.5}]}
    cache.store("identify", "deadbeef", payload)
    assert cache.load("identify", "deadbeef") == payload


def test_two_kinds_do_not_collide():
    cache.store("identify", "ab", {"a": 1})
    cache.store("derive", "ab", {"a": 2})
    assert cache.load("identify", "ab") == {"a": 1}
    assert cache.load("derive", "ab") == {"a": 2}


def test_the_same_runspec_produces_the_same_key():
    a = RunSpec(DEFAULT_RUN.plant, DEFAULT_RUN.drift, ProtocolSpec(T_log_ms=5))
    b = RunSpec(DEFAULT_RUN.plant, DEFAULT_RUN.drift, ProtocolSpec(T_log_ms=5.0))
    assert run_hash(a) == run_hash(b)
    cache.store("identify", run_hash(a), {"value": 1})
    assert cache.load("identify", run_hash(b)) == {"value": 1}


def test_a_changed_runspec_misses():
    cache.store("identify", run_hash(DEFAULT_RUN), {"value": 1})
    other = RunSpec(DEFAULT_RUN.plant, DEFAULT_RUN.drift, ProtocolSpec(seed=99))
    assert cache.load("identify", run_hash(other)) is None


def test_corrupt_entry_is_a_miss_not_a_crash():
    path = cache.cache_path("identify", "bad")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert cache.load("identify", "bad") is None


def test_clear_removes_only_the_named_kind():
    cache.store("identify", "a", {"v": 1})
    cache.store("derive", "b", {"v": 2})
    assert cache.clear("identify") == 1
    assert cache.load("identify", "a") is None
    assert cache.load("derive", "b") == {"v": 2}


def test_non_hex_keys_are_rejected():
    with pytest.raises(ValueError):
        cache.cache_path("identify", "zzz")


def test_clear_with_traversal_kind_raises_valueerror():
    with pytest.raises(ValueError):
        cache.clear("../../..")


def test_concurrent_stores_do_not_collide_on_temp_path():
    # Two sequential stores to the same key should not conflict on temp names
    cache.store("identify", "ab", {"a": 1})
    cache.store("identify", "ab", {"a": 2})
    # After both stores complete and the final one wins, there should be no .tmp files
    cache_root = cache.CACHE_DIR / "identify"
    tmp_files = list(cache_root.glob("*.tmp"))
    assert len(tmp_files) == 0
    # Final value should be the second store
    assert cache.load("identify", "ab") == {"a": 2}


def test_orphaned_temp_files_are_removed_by_clear():
    # Create a cache file
    cache.store("identify", "ab", {"v": 1})
    # Manually create an orphaned temp file (simulating a crash)
    cache_path = cache.CACHE_DIR / "identify" / "ab.json"
    orphaned_temp = cache_path.with_name(f"{cache_path.name}.12345.abcd1234.tmp")
    orphaned_temp.write_text("{incomplete", encoding="utf-8")
    # clear() should remove both the cache file and the orphaned temp
    assert cache.clear("identify") == 1  # Only the actual cache file is counted
    # Verify both the cache file and the temp file are gone
    assert not cache_path.exists()
    assert not orphaned_temp.exists()
