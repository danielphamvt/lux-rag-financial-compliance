"""Tests for the SQLite chat history store (pure, LLM-free)."""
import pytest

from db import chat_store


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_store, "DB_PATH", str(tmp_path / "chat_history.db"))


def test_roundtrip_and_ordering():
    cid = chat_store.create_conversation("title")
    chat_store.add_message(cid, "user", "hello", plain="hello")
    chat_store.add_message(cid, "assistant", "<b>hi</b>", plain="hi")

    msgs = chat_store.get_messages(cid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "hello"
    assert msgs[1]["content"] == "<b>hi</b>"

    assert chat_store.plain_messages(cid) == [("user", "hello"), ("assistant", "hi")]
    assert chat_store.list_recent() == [("title", cid)]


def test_make_title():
    assert chat_store.make_title("/search_documents What is the fee?") == "What is the fee?"
    assert chat_store.make_title("  Establishment   conditions\noverview  ") == "Establishment conditions overview"
    assert chat_store.make_title("   ") == "New chat"

    long = "The obligations described for organizations regarding quarterly reporting "
    title = chat_store.make_title(long)
    assert title.endswith("…")
    assert len(title) <= 49  # 48-char cut + ellipsis
    assert title.startswith("The obligations described")


def test_update_bumps_recency():
    import time

    first = chat_store.create_conversation("first")
    second = chat_store.create_conversation("second")
    # Timestamps have millisecond resolution: step past it so the bump below
    # is strictly newer than `second`'s creation
    time.sleep(0.002)
    chat_store.add_message(first, "user", "q", plain="q")
    # `first` was updated after `second` was created, so it is the most recent
    assert chat_store.list_recent()[0] == ("first", first)


def test_prune_deletes_oldest_conversation(monkeypatch):
    monkeypatch.delenv("MAX_STORED_CONVERSATIONS", raising=False)
    assert chat_store._keep_count() == 20

    monkeypatch.setenv("MAX_STORED_CONVERSATIONS", "2")
    oldest = chat_store.create_conversation("oldest")
    chat_store.add_message(oldest, "user", "q", plain="q")
    middle = chat_store.create_conversation("middle")
    newest = chat_store.create_conversation("newest")  # pushes `oldest` out

    kept = [cid for _, cid in chat_store.list_recent()]
    assert kept == [newest, middle]
    # Cascade removed the pruned conversation's messages
    assert chat_store.get_messages(oldest) == []
    assert chat_store.plain_messages(oldest) == []


def test_delete_conversation():
    cid = chat_store.create_conversation("to-delete")
    chat_store.add_message(cid, "user", "q", plain="q")
    chat_store.delete_conversation(cid)
    assert chat_store.list_recent() == []
    assert chat_store.get_messages(cid) == []
    assert chat_store.plain_messages(cid) == []
