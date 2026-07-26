import asyncio
import json

import llm
import pytest


@pytest.fixture(scope="session")
def apple_model_is_available():
    fm = pytest.importorskip("apple_fm_sdk")
    available, reason = fm.SystemLanguageModel().is_available()
    if not available:
        pytest.skip(f"Apple Foundation model is unavailable: {reason}")


@pytest.fixture
def model(apple_model_is_available):
    return llm.get_model("apple-foundation")


def test_models_are_registered():
    assert llm.get_model("apple-foundation").model_id == "apple-foundation"
    assert llm.get_model("afm").model_id == "apple-foundation"
    assert llm.get_async_model("apple-foundation").model_id == "apple-foundation"


def test_options_are_registered():
    options = llm.get_model("apple-foundation").Options.model_json_schema()[
        "properties"
    ]
    assert set(options) == {
        "temperature",
        "max_tokens",
        "sampling",
        "top",
        "probability_threshold",
        "seed",
        "use_case",
        "guardrails",
    }


def test_prompt_uses_actual_model(model):
    response = model.prompt(
        "Reply with a brief greeting.",
        system="You answer concisely.",
        stream=False,
        sampling="greedy",
        max_tokens=20,
    )

    assert response.text().strip()
    assert response.json()["type"] == "FoundationModels.Transcript"


def test_streaming_uses_actual_model_and_returns_deltas(model):
    response = model.prompt(
        "Write a paragraph of about eighty words describing seabirds.",
        stream=True,
        sampling="greedy",
        max_tokens=150,
    )

    chunks = list(response)
    text = "".join(chunks)
    assert len(chunks) > 1
    assert text.strip()
    # apple-fm-sdk streams cumulative snapshots. A second raw snapshot would
    # start with the entire first snapshot; a correctly converted delta does not.
    assert not chunks[1].startswith(chunks[0])
    assert response.text() == text
    assert response.json()["type"] == "FoundationModels.Transcript"


def test_schema_uses_actual_model(model):
    schema = {
        "type": "object",
        "properties": {
            "bird": {"type": "string"},
            "legs": {"type": "integer"},
        },
        "required": ["bird", "legs"],
    }
    response = model.prompt(
        "Describe a pelican.",
        schema=schema,
        stream=False,
        sampling="greedy",
        max_tokens=30,
    )

    data = json.loads(response.text())
    assert isinstance(data["bird"], str)
    assert isinstance(data["legs"], int)
    assert response.json()["type"] == "FoundationModels.Transcript"


def test_conversation_uses_actual_model_and_restores_transcript(model):
    conversation = model.conversation()
    first = conversation.prompt(
        "Remember the phrase 'violet pelican'. Reply only with OK.",
        stream=False,
        sampling="greedy",
        max_tokens=10,
    )
    assert first.text().strip()
    assert first.json()["type"] == "FoundationModels.Transcript"

    second = conversation.prompt(
        "What phrase did I ask you to remember?",
        stream=False,
        sampling="greedy",
        max_tokens=20,
    )
    answer = second.text().lower()
    assert "violet" in answer
    assert "pelican" in answer


def test_async_prompt_uses_actual_model(apple_model_is_available):
    async def run_prompt():
        model = llm.get_async_model("apple-foundation")
        response = model.prompt(
            "Reply with one short greeting.",
            stream=False,
            sampling="greedy",
            max_tokens=20,
        )
        assert (await response.text()).strip()
        assert (await response.json())["type"] == "FoundationModels.Transcript"

    asyncio.run(run_prompt())
