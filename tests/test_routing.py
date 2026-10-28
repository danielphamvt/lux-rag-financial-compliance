"""Tests for the greeting router predicate (no LLM / no embeddings needed)."""
import unicodedata

from rag_agent.routing import (
    is_greeting_message,
    is_out_of_domain_message,
    normalize_message,
)


def test_short_legal_question_is_not_a_greeting():
    # Words like "which" or "this" contain "hi" and must not be misrouted by substring matching.
    assert not is_greeting_message("Which circular applies?")
    assert not is_greeting_message("What is the NAV error tolerance threshold?")
    assert not is_greeting_message("Is indemnification required for UCITS?")


def test_nfd_decomposed_input_is_recognized():
    nfd_greeting = unicodedata.normalize("NFD", "hello")
    assert normalize_message(nfd_greeting) == "hello"
    assert is_greeting_message(nfd_greeting)


def test_greetings_and_farewells_are_recognized():
    for text in ["hi", "Hello!", "HELLO", "thank you", "thanks", "bye", "goodbye"]:
        assert is_greeting_message(text), text


def test_messages_with_extra_words_are_not_greetings():
    # Small-talk plus an actual question, as well as empty/None inputs, must not be treated as greetings.
    assert not is_greeting_message("thank you, but what is the threshold?")
    assert not is_greeting_message("hi, what does Table 10 require?")
    assert not is_greeting_message("")
    assert not is_greeting_message(None)


def test_tier1_out_of_domain_messages_are_detected_and_routed_to_direct_answer():
    from langchain_core.messages import HumanMessage
    from rag_agent.nodes_edges import is_greeting
    from rag_agent.query_rewriter import direct_answer

    off_topic_queries = [
        "How do I cook beef pho and make pasta dough from scratch?",
        "Who won the Champions League football match yesterday?",
        "What is the weather forecast and temperature in Paris tomorrow?",
        "Write a Python function to reverse a binary tree on LeetCode",
        "/all_sources Write a haiku poem about summer rain",
    ]
    for q in off_topic_queries:
        assert is_out_of_domain_message(q), f"Expected out-of-domain True for: {q}"
        state = {"messages": [HumanMessage(content=q)]}
        assert is_greeting(state) == "direct_answer"
        out = direct_answer(state)
        assert "⚠️ **Out of Scope:**" in out["messages"][0].content

    in_domain_queries = [
        "What is the NAV error tolerance threshold for Money Market Funds under CSSF 24/856?",
        "What are the AIFMD leverage reporting requirements?",
        "How does MiFID II inducements rule compare to UCITS?",
        "Under Circular CSSF 04/146, what are the market timing rules?",
        "What about Class R?",
        "And for 2025?",
        "hi",
        "",
    ]
    for q in in_domain_queries:
        assert not is_out_of_domain_message(q), f"Expected out-of-domain False for: {q}"
