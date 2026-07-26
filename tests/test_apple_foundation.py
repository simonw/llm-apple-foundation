import asyncio
import json
import re
from importlib.metadata import version

import llm
import pytest
from click.testing import CliRunner
from llm.cli import cli
from pydantic import ValidationError


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
    model = llm.get_model("apple-foundation")
    async_model = llm.get_async_model("apple-foundation")
    assert model.model_id == "apple-foundation"
    assert llm.get_model("afm").model_id == "apple-foundation"
    assert async_model.model_id == "apple-foundation"
    assert model.supports_tools
    assert async_model.supports_tools


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


@pytest.mark.parametrize(
    "options",
    (
        {"top": 10},
        {
            "sampling": "random",
            "top": 10,
            "probability_threshold": 0.9,
        },
    ),
)
def test_invalid_sampling_option_combinations(options):
    model = llm.get_model("apple-foundation")
    with pytest.raises(ValidationError):
        model.prompt("Hello", **options)


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


def test_conversation_survives_database_round_trip(apple_model_is_available, tmp_path):
    database = str(tmp_path / "logs.db")
    runner = CliRunner()
    first = runner.invoke(
        cli,
        (
            "-m",
            "apple-foundation",
            "--no-stream",
            "--database",
            database,
            "Remember the phrase 'amber puffin'. Reply only with OK.",
        ),
    )
    assert first.exit_code == 0, first.output

    second = runner.invoke(
        cli,
        (
            "-c",
            "--no-stream",
            "--database",
            database,
            "What phrase did I ask you to remember?",
        ),
    )
    assert second.exit_code == 0, second.output
    assert "amber" in second.output.lower()
    assert "puffin" in second.output.lower()


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


def test_tool_uses_actual_model(model):
    marker = "apple-foundation-tool-verification-7f31"
    calls = []

    def verification_tool() -> str:
        "Return the verification marker requested by the user."
        calls.append(True)
        return marker

    response = model.prompt(
        "Call verification_tool and reply with its exact result.",
        tools=[verification_tool],
        stream=False,
        sampling="greedy",
        max_tokens=100,
    )

    text = response.text()
    assert calls == [True]
    assert marker in text


def test_typed_async_tool_uses_actual_model(model):
    calls = []

    async def join_words(left: str, right: str) -> str:
        "Join the two supplied words with the verification separator."
        calls.append((left, right))
        return f"{left}::verified::{right}"

    response = model.prompt(
        "Call join_words with left set to 'violet' and right set to 'pelican'. "
        "Reply with its exact result.",
        tools=[join_words],
        stream=False,
        sampling="greedy",
        max_tokens=100,
    )

    text = response.text()
    assert calls == [("violet", "pelican")]
    assert "violet::verified::pelican" in text


def test_async_model_tool_uses_actual_model(apple_model_is_available):
    marker = "async-apple-tool-verification-42c9"
    calls = []

    async def run_prompt():
        def async_model_tool() -> str:
            "Return the async model verification marker."
            calls.append(True)
            return marker

        model = llm.get_async_model("apple-foundation")
        response = model.prompt(
            "Call async_model_tool and reply with its exact result.",
            tools=[async_model_tool],
            stream=False,
            sampling="greedy",
            max_tokens=100,
        )
        assert marker in await response.text()

    asyncio.run(run_prompt())
    assert calls == [True]


def test_llm_time_tool_works_from_cli(apple_model_is_available, tmp_path):
    result = CliRunner().invoke(
        cli,
        (
            "-m",
            "afm",
            "-T",
            "llm_time",
            "--database",
            str(tmp_path / "logs.db"),
            "what time is it?",
        ),
    )

    assert result.exit_code == 0, result.output
    assert re.search(r"\b\d{1,2}:\d{2}\b", result.output), result.output


def test_tools_debug_uses_llm_lifecycle_hooks(apple_model_is_available, tmp_path):
    result = CliRunner().invoke(
        cli,
        (
            "-m",
            "afm",
            "-T",
            "llm_version",
            "--td",
            "--database",
            str(tmp_path / "logs.db"),
            "Call llm_version and report its exact result.",
        ),
    )

    assert result.exit_code == 0, result.output
    assert "Tool call: llm_version({})" in result.output
    assert version("llm") in result.output


def test_tool_uses_llm_lifecycle_hooks(model):
    marker = "apple-foundation-lifecycle-verification-85d2"
    events = []

    def lifecycle_tool(llm_tool_call: llm.ToolCall) -> str:
        "Return the lifecycle verification marker."
        events.append(("implementation", llm_tool_call.tool_call_id))
        return marker

    def before_call(tool, tool_call):
        events.append(("before", tool.name, tool_call.tool_call_id))

    def after_call(tool, tool_call, tool_result):
        events.append(
            (
                "after",
                tool.name,
                tool_call.tool_call_id,
                tool_result.output,
            )
        )

    response = model.chain(
        "Call lifecycle_tool exactly once and reply with its exact result.",
        tools=[lifecycle_tool],
        before_call=before_call,
        after_call=after_call,
        stream=False,
        options={"sampling": "greedy", "max_tokens": 100},
    )

    assert marker in response.text()
    tool_call_id = events[0][2]
    assert tool_call_id.startswith("tc_")
    assert events == [
        ("before", "lifecycle_tool", tool_call_id),
        ("implementation", tool_call_id),
        ("after", "lifecycle_tool", tool_call_id, marker),
    ]
