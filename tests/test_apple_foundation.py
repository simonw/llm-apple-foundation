import asyncio
import json
import re
from importlib.metadata import version

import llm
import llm_apple_foundation
import pytest
from click.testing import CliRunner
from llm.cli import cli
from llm.parts import (
    AttachmentPart,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from pydantic import ValidationError

GREEDY = {"sampling": "greedy"}


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
    transcript = response.json()
    assert transcript["type"] == "FoundationModels.Transcript"
    roles = [entry["role"] for entry in transcript["transcript"]["entries"]]
    # Only this turn's entries are recorded - the instructions entry and any
    # earlier history are already part of prompt.messages
    assert roles == ["user", "response"]


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


# Converting llm.Message history into Apple transcripts


def _roles(entries):
    return [
        entry["role"] + ("[toolCalls]" if "toolCalls" in entry else "")
        for entry in entries
    ]


def test_split_messages_separates_instructions_history_and_prompt():
    turn = llm_apple_foundation._split_messages(
        [
            llm.system("Be terse."),
            llm.user("one"),
            llm.assistant("two"),
            llm.user("three", "four"),
        ]
    )
    assert turn.instructions == "Be terse."
    assert [message.role for message in turn.history] == ["user", "assistant"]
    assert turn.prompt == ["three", "four"]


def test_split_messages_without_trailing_user_message():
    turn = llm_apple_foundation._split_messages(
        [
            llm.user("What is the code?"),
            llm.assistant(ToolCallPart(name="code", tool_call_id="tc_1")),
            llm.tool_message(
                ToolResultPart(name="code", output="4417", tool_call_id="tc_1")
            ),
        ]
    )
    assert turn.instructions is None
    assert len(turn.history) == 3
    assert turn.prompt == ""


def test_transcript_builder_converts_message_parts():
    own_metadata = {"apple_foundation": {"provider_managed": True}}
    builder = llm_apple_foundation._TranscriptBuilder()
    for message in [
        llm.user("Look up the code"),
        llm.assistant(
            ReasoningPart(text="Apple has no reasoning entries"),
            TextPart(text="Checking."),
            ToolCallPart(name="lookup", arguments={"key": "a"}, tool_call_id="tc_1"),
        ),
        llm.tool_message(
            ToolResultPart(name="lookup", output="4417", tool_call_id="tc_1")
        ),
        llm.assistant(
            ToolCallPart(
                name="secret",
                tool_call_id="tc_2",
                server_executed=True,
                provider_metadata=own_metadata,
            ),
            ToolResultPart(
                name="secret",
                output="marmalade",
                tool_call_id="tc_2",
                server_executed=True,
                provider_metadata=own_metadata,
            ),
            # Another provider's server-side tool cannot be replayed
            ToolCallPart(name="web_search", tool_call_id="ws_1", server_executed=True),
            ToolResultPart(
                name="web_search",
                output="results",
                tool_call_id="ws_1",
                server_executed=True,
            ),
            TextPart(text="The code is 4417."),
        ),
    ]:
        builder.add_message(message)

    entries = builder.entries
    assert _roles(entries) == [
        "user",
        "response",
        "response[toolCalls]",
        "tool",
        "response[toolCalls]",
        "tool",
        "response",
    ]
    assert entries[0]["options"] == {}
    assert entries[0]["contents"][0]["text"] == "Look up the code"
    assert entries[1]["contents"][0]["text"] == "Checking."
    assert entries[2]["toolCalls"] == [
        {"id": "tc_1", "name": "lookup", "arguments": '{"key": "a"}'}
    ]
    assert entries[3] == {
        "role": "tool",
        "id": "tc_1",
        "toolCallID": "tc_1",
        "toolName": "lookup",
        "contents": [
            {"type": "text", "id": entries[3]["contents"][0]["id"], "text": "4417"}
        ],
    }
    assert entries[4]["toolCalls"][0]["name"] == "secret"
    assert entries[5]["contents"][0]["text"] == "marmalade"
    assert entries[6]["contents"][0]["text"] == "The code is 4417."
    # Every entry and segment needs an id for Transcript.from_dict()
    for entry in entries:
        assert entry["id"]
        for segment in entry.get("contents", []):
            assert segment["id"]


def test_transcript_builder_rejects_attachments():
    builder = llm_apple_foundation._TranscriptBuilder()
    with pytest.raises(llm.ModelError, match="attachments"):
        builder.add_message(
            llm.user(
                "Describe this",
                AttachmentPart(
                    attachment=llm.Attachment(content=b"x", type="image/png")
                ),
            )
        )


def test_built_transcript_loads_in_apple_sdk():
    fm = pytest.importorskip("apple_fm_sdk")
    builder = llm_apple_foundation._TranscriptBuilder()
    for message in [
        llm.user("Look up the code"),
        llm.assistant(ToolCallPart(name="lookup", tool_call_id="tc_1")),
        llm.tool_message(
            ToolResultPart(name="lookup", output="4417", tool_call_id="tc_1")
        ),
        llm.assistant("The code is 4417."),
    ]:
        builder.add_message(message)

    async def round_trip():
        transcript = await fm.Transcript.from_dict(
            {
                "version": 1,
                "type": "FoundationModels.Transcript",
                "transcript": {"entries": builder.entries},
            }
        )
        return await transcript.to_dict()

    entries = asyncio.run(round_trip())["transcript"]["entries"]
    assert _roles(entries) == ["user", "response[toolCalls]", "tool", "response"]


def test_model_variant_is_recorded_as_resolved_model():
    class Variant:
        display_name = "AFM 3 Core"

    class SystemModel:
        def __init__(self, variant):
            self.variant = variant

    class Response:
        resolved_model = None

        def set_resolved_model(self, model_id):
            self.resolved_model = model_id

    response = Response()
    llm_apple_foundation._record_model_variant(SystemModel(None), response)
    assert response.resolved_model is None
    llm_apple_foundation._record_model_variant(SystemModel(Variant()), response)
    assert response.resolved_model == "AFM 3 Core"


# Conversations against the actual model


def test_messages_history_uses_actual_model(model):
    response = model.prompt(
        messages=[
            llm.user("Remember the phrase 'violet pelican'. Reply only with OK."),
            llm.assistant("OK"),
            llm.user("What phrase did I ask you to remember?"),
        ],
        stream=False,
        max_tokens=30,
        **GREEDY,
    )
    answer = response.text().lower()
    assert "violet" in answer
    assert "pelican" in answer


def test_reply_continues_from_response(model):
    first = model.prompt(
        "Remember the phrase 'saffron gannet'. Reply only with OK.",
        stream=False,
        max_tokens=10,
        **GREEDY,
    )
    first.text()
    second = first.reply("What phrase did I ask you to remember?")
    answer = second.text().lower()
    assert "saffron" in answer
    assert "gannet" in answer


def test_system_prompt_persists_across_turns(model):
    conversation = model.conversation()
    conversation.prompt(
        "Say hello.",
        system="You are the assistant for a bird sanctuary called Tangerine Cove.",
        stream=False,
        max_tokens=20,
        **GREEDY,
    ).text()
    second = conversation.prompt(
        "What is the name of the bird sanctuary you work for?",
        stream=False,
        max_tokens=30,
        **GREEDY,
    )
    assert "tangerine cove" in second.text().lower()
    entries = second.json()["transcript"]["entries"]
    assert [entry["role"] for entry in entries] == ["user", "response"]


def test_tool_calls_are_recorded_and_replayed(model):
    calls = []

    def lookup_bird_record() -> str:
        "Look up the bird record."
        calls.append(True)
        return "Record: species=puffin, ring=K4417, colony=Skomer"

    conversation = model.conversation(tools=[lookup_bird_record])
    first = conversation.prompt(
        "Call lookup_bird_record, then reply with only the word DONE.",
        stream=True,
        max_tokens=40,
        **GREEDY,
    )
    first.text()
    assert calls == [True]

    # The provider-managed call is recorded in the response, marked as
    # already executed so that LLM does not run it again
    parts = first.messages()[0].parts
    tool_call = next(part for part in parts if isinstance(part, ToolCallPart))
    tool_result = next(part for part in parts if isinstance(part, ToolResultPart))
    assert tool_call.name == "lookup_bird_record"
    assert tool_call.server_executed
    assert tool_call.provider_metadata == {
        "apple_foundation": {"provider_managed": True}
    }
    assert tool_result.tool_call_id == tool_call.tool_call_id
    assert tool_result.output == "Record: species=puffin, ring=K4417, colony=Skomer"
    assert first.tool_calls() == []
    roles = _roles(first.json()["transcript"]["entries"])
    assert roles == ["user", "response[toolCalls]", "tool", "response"]

    second = conversation.prompt(
        "Without calling any tools: what ring number did the bird record show?",
        stream=False,
        max_tokens=40,
        **GREEDY,
    )
    assert "k4417" in second.text().lower()
    assert calls == [True]


BIRD_RECORD_FUNCTION = """
def lookup_bird_record() -> str:
    "Look up the bird record."
    return "Record: species=puffin, ring=K4417, colony=Skomer"
"""


def test_tool_history_survives_database_round_trip(apple_model_is_available, tmp_path):
    database = str(tmp_path / "logs.db")
    runner = CliRunner()
    first = runner.invoke(
        cli,
        (
            "-m",
            "afm",
            "--functions",
            BIRD_RECORD_FUNCTION,
            "-o",
            "sampling",
            "greedy",
            "--database",
            database,
            "Call lookup_bird_record, then reply with only the word DONE.",
        ),
    )
    assert first.exit_code == 0, first.output
    assert "k4417" not in first.output.lower()

    second = runner.invoke(
        cli,
        (
            "-c",
            "-o",
            "sampling",
            "greedy",
            "--database",
            database,
            "What ring number did the bird record show?",
        ),
    )
    assert second.exit_code == 0, second.output
    assert "k4417" in second.output.lower()

    logs = runner.invoke(cli, ("logs", "-c", "--json", "--database", database))
    assert logs.exit_code == 0, logs.output
    rows = json.loads(logs.output)
    assert [row["tool_calls"][0]["name"] for row in rows[:1]] == ["lookup_bird_record"]
    assert rows[0]["tool_calls"][0]["server_executed"]
    assert rows[0]["tool_results"][0]["output"] == (
        "Record: species=puffin, ring=K4417, colony=Skomer"
    )
    assert rows[1]["tool_calls"] == []


def test_client_tool_history_from_another_model(model):
    response = model.prompt(
        messages=[
            llm.user("What is the weather in Lisbon?"),
            llm.assistant(
                ToolCallPart(
                    name="get_weather",
                    arguments={"city": "Lisbon"},
                    tool_call_id="call_1",
                )
            ),
            llm.tool_message(
                ToolResultPart(
                    name="get_weather",
                    output="Lisbon: 23C, light breeze from the northwest",
                    tool_call_id="call_1",
                )
            ),
            llm.assistant("It is 23C in Lisbon."),
            llm.user("Which direction was the breeze coming from?"),
        ],
        stream=False,
        max_tokens=30,
        **GREEDY,
    )
    assert "northwest" in response.text().lower()


class ScriptedModel(llm.Model):
    model_id = "scripted-test-model"

    def execute(self, prompt, stream, response, conversation=None):
        yield "I have assigned you the codename 'copper kestrel'."


class ScriptedModelPlugin:
    __name__ = "ScriptedModelPlugin"

    @llm.hookimpl
    def register_models(self, register):
        register(ScriptedModel())


@pytest.fixture
def scripted_model():
    llm.plugins.pm.register(ScriptedModelPlugin(), name="scripted-test-model")
    try:
        yield
    finally:
        llm.plugins.pm.unregister(name="scripted-test-model")


def test_continue_conversation_started_with_another_model(
    apple_model_is_available, scripted_model, tmp_path
):
    database = str(tmp_path / "logs.db")
    runner = CliRunner()
    first = runner.invoke(
        cli,
        (
            "-m",
            "scripted-test-model",
            "--database",
            database,
            "Please give me a codename.",
        ),
    )
    assert first.exit_code == 0, first.output

    second = runner.invoke(
        cli,
        (
            "-c",
            "-m",
            "afm",
            "-o",
            "sampling",
            "greedy",
            "--database",
            database,
            "What codename did you assign me?",
        ),
    )
    assert second.exit_code == 0, second.output
    assert "copper kestrel" in second.output.lower()


def test_schema_response_in_conversation_history(model):
    conversation = model.conversation()
    first = conversation.prompt(
        "Invent a bird with a name and a wingspan in centimeters.",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "wingspan_cm": {"type": "integer"},
            },
            "required": ["name", "wingspan_cm"],
        },
        stream=False,
        max_tokens=60,
        **GREEDY,
    )
    bird = json.loads(first.text())
    second = conversation.prompt(
        "What was the name of the bird you invented? Reply with just the name.",
        stream=False,
        max_tokens=30,
        **GREEDY,
    )
    assert bird["name"].lower() in second.text().lower()


def test_async_conversation_uses_actual_model(apple_model_is_available):
    async def run_conversation():
        model = llm.get_async_model("apple-foundation")
        conversation = model.conversation()
        first = conversation.prompt(
            "Remember the phrase 'cobalt shearwater'. Reply only with OK.",
            stream=False,
            max_tokens=10,
            **GREEDY,
        )
        await first.text()
        second = conversation.prompt(
            "What phrase did I ask you to remember?",
            stream=True,
            max_tokens=30,
            **GREEDY,
        )
        return await second.text()

    answer = asyncio.run(run_conversation()).lower()
    assert "cobalt" in answer
    assert "shearwater" in answer


def test_async_streaming_tool_call_is_recorded(apple_model_is_available):
    calls = []

    async def run_prompt():
        async def lookup_marker() -> str:
            "Return the verification marker."
            calls.append(True)
            return "async-stream-marker-5521"

        model = llm.get_async_model("apple-foundation")
        response = model.prompt(
            "Call lookup_marker and reply with its exact result.",
            tools=[lookup_marker],
            stream=True,
            max_tokens=40,
            **GREEDY,
        )
        chunks = [chunk async for chunk in response]
        return chunks, await response.messages(), await response.tool_calls()

    chunks, messages, tool_calls = asyncio.run(run_prompt())
    assert calls == [True]
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert "async-stream-marker-5521" in "".join(chunks)
    assert tool_calls == []
    parts = messages[0].parts
    assert [type(part) for part in parts] == [ToolCallPart, ToolResultPart, TextPart]
    assert parts[1].output == "async-stream-marker-5521"


def test_attachments_in_history_are_rejected(model):
    with pytest.raises(llm.ModelError, match="attachments"):
        model.prompt(
            messages=[
                llm.user(
                    "Describe this image",
                    AttachmentPart(
                        attachment=llm.Attachment(content=b"x", type="image/png")
                    ),
                ),
                llm.assistant("It is a pelican."),
                llm.user("What was it?"),
            ],
            stream=False,
        ).text()


def test_context_window_error_is_explained(model):
    sentence = "The brown pelican dives from great heights to catch fish. "
    with pytest.raises(llm.ModelError, match=r"context window of \d+ tokens"):
        model.prompt(
            messages=[
                llm.user(sentence * 400),
                llm.assistant("OK"),
                llm.user("Summarize that."),
            ],
            stream=False,
        ).text()
