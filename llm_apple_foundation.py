import asyncio
import copy
import json
import queue
import threading
import uuid
from dataclasses import dataclass
from typing import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Iterator,
    Literal,
    Optional,
    Union,
)

import llm
from llm.parts import (
    AttachmentPart,
    StreamEvent,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from pydantic import Field, model_validator

ToolExecutor = Callable[[llm.ToolCall], Awaitable[llm.ToolResult]]

# Namespace for provider_metadata on parts this plugin records. Tool calls
# that Apple's session runs mid-response are tagged with it so they can be
# replayed into the transcript when the conversation continues.
PROVIDER = "apple_foundation"


@dataclass
class _ToolExecutionRequest:
    tool_call: llm.ToolCall
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future


class AppleFoundationOptions(llm.Options):
    temperature: Optional[float] = Field(
        description="Sampling temperature; higher values produce more varied output",
        default=None,
        ge=0,
    )
    max_tokens: Optional[int] = Field(
        description="Maximum number of tokens to generate",
        default=None,
        gt=0,
    )
    sampling: Optional[Literal["greedy", "random"]] = Field(
        description="Token sampling strategy",
        default=None,
    )
    top: Optional[int] = Field(
        description="For random sampling, consider only the top-k tokens",
        default=None,
        gt=0,
    )
    probability_threshold: Optional[float] = Field(
        description="For random sampling, cumulative probability threshold",
        default=None,
        ge=0,
        le=1,
    )
    seed: Optional[int] = Field(
        description="Random seed for reproducible random sampling",
        default=None,
    )
    use_case: Literal["general", "content_tagging"] = Field(
        description="Apple system model use case",
        default="general",
    )
    guardrails: Literal["default", "permissive_content_transformations"] = Field(
        description="Apple system model guardrail configuration",
        default="default",
    )

    @model_validator(mode="after")
    def validate_sampling_options(self):
        if self.top is not None and self.probability_threshold is not None:
            raise ValueError("top and probability_threshold cannot be used together")
        if (
            self.top is not None
            or self.probability_threshold is not None
            or self.seed is not None
        ) and self.sampling != "random":
            raise ValueError(
                "top, probability_threshold, and seed require sampling=random"
            )
        return self


def _load_sdk():
    try:
        import apple_fm_sdk as fm
    except ImportError as ex:
        raise llm.ModelError(
            "apple-fm-sdk is required and is only available on macOS"
        ) from ex
    return fm


def _create_system_model(fm, options):
    use_cases = {
        "general": fm.SystemLanguageModelUseCase.GENERAL,
        "content_tagging": fm.SystemLanguageModelUseCase.CONTENT_TAGGING,
    }
    guardrails = {
        "default": fm.SystemLanguageModelGuardrails.DEFAULT,
        "permissive_content_transformations": (
            fm.SystemLanguageModelGuardrails.PERMISSIVE_CONTENT_TRANSFORMATIONS
        ),
    }
    model = fm.SystemLanguageModel(
        use_case=use_cases[options.use_case],
        guardrails=guardrails[options.guardrails],
    )
    available, reason = model.is_available()
    if not available:
        reason_name = getattr(reason, "name", str(reason))
        raise llm.ModelError(f"Apple Foundation model is unavailable: {reason_name}")
    return model


def _record_model_variant(system_model, response):
    # Only reported on macOS 27+ with an SDK built against the macOS 27 SDK
    variant = system_model.variant
    if variant is not None and variant.display_name:
        response.set_resolved_model(variant.display_name)


def _create_generation_options(fm, options):
    sampling = None
    if options.sampling == "greedy":
        sampling = fm.SamplingMode.greedy()
    elif options.sampling == "random":
        sampling = fm.SamplingMode.random(
            top=options.top,
            probability_threshold=options.probability_threshold,
            seed=options.seed,
        )
    return fm.GenerationOptions(
        sampling=sampling,
        temperature=options.temperature,
        maximum_response_tokens=options.max_tokens,
    )


def _schema_title(name):
    return "".join(part.capitalize() for part in name.replace("-", "_").split("_"))


def _normalize_schema(schema):
    normalized = copy.deepcopy(schema)

    def visit(node, title):
        if not isinstance(node, dict):
            return

        properties = node.get("properties")
        if node.get("type") == "object" or properties is not None:
            node.setdefault("type", "object")
            node.setdefault("title", title)
            node.setdefault("additionalProperties", False)
            properties = properties or {}
            node["x-order"] = list(properties)
            for name, child in properties.items():
                visit(child, _schema_title(name))

        for name, child in node.get("$defs", {}).items():
            visit(child, name)

        if "items" in node:
            visit(node["items"], f"{title}Item")

        for keyword in ("allOf", "anyOf", "oneOf"):
            for index, child in enumerate(node.get(keyword, [])):
                visit(child, f"{title}{index + 1}")

    visit(normalized, normalized.get("title", "Response"))
    return normalized


def _tool_schema(fm, tool):
    from apple_fm_sdk.generation_property import Property

    root_schema = tool.input_schema or {"type": "object", "properties": {}}
    definitions = {
        **root_schema.get("$defs", {}),
        **root_schema.get("definitions", {}),
    }

    def schema_error(message):
        raise llm.ModelError(
            f"Tool '{tool.name}' has an unsupported input schema: {message}"
        )

    def resolve(schema):
        reference = schema.get("$ref")
        if not reference:
            return schema
        prefix, _, name = reference.rpartition("/")
        if prefix not in ("#/$defs", "#/definitions") or name not in definitions:
            schema_error(f"unresolved reference {reference!r}")
        return definitions[name]

    def guides_for(schema):
        guides = []
        enum = schema.get("enum")
        if enum is not None:
            if not enum or not all(isinstance(value, str) for value in enum):
                schema_error("enum values must be strings")
            guides.append(fm.GenerationGuide.anyOf(enum))
        if "const" in schema:
            if not isinstance(schema["const"], str):
                schema_error("const values must be strings")
            guides.append(fm.GenerationGuide.constant(schema["const"]))
        if "minimum" in schema:
            guides.append(fm.GenerationGuide.minimum(schema["minimum"]))
        if "maximum" in schema:
            guides.append(fm.GenerationGuide.maximum(schema["maximum"]))
        if "minItems" in schema:
            guides.append(fm.GenerationGuide.min_items(schema["minItems"]))
        if "maxItems" in schema:
            guides.append(fm.GenerationGuide.max_items(schema["maxItems"]))
        if "pattern" in schema:
            guides.append(fm.GenerationGuide.regex(schema["pattern"]))
        return guides

    def field_type(schema, title):
        schema = resolve(schema)
        variants = schema.get("anyOf") or schema.get("oneOf")
        if variants:
            non_null = [
                variant
                for variant in variants
                if resolve(variant).get("type") != "null"
            ]
            if len(non_null) != 1 or len(non_null) == len(variants):
                schema_error("only nullable anyOf/oneOf unions are supported")
            value_type, nested = field_type(non_null[0], title)
            return Optional[value_type], nested

        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            non_null_types = [item for item in schema_type if item != "null"]
            if len(non_null_types) != 1 or len(non_null_types) == len(schema_type):
                schema_error("only nullable type unions are supported")
            nullable_schema = dict(schema)
            nullable_schema["type"] = non_null_types[0]
            value_type, nested = field_type(nullable_schema, title)
            return Optional[value_type], nested

        primitive_types = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
        }
        if schema_type in primitive_types:
            return primitive_types[schema_type], []
        if schema_type == "array":
            items = schema.get("items")
            if not isinstance(items, dict):
                schema_error("arrays must declare a single items schema")
            item_type, nested = field_type(items, f"{title}Item")
            return list[item_type], nested
        if schema_type == "object" or "properties" in schema:
            nested_schema = build_schema(schema, title)
            return nested_schema.type_class, [
                nested_schema,
                *nested_schema.dynamic_nested_types,
            ]
        schema_error(f"unknown type {schema_type!r}")

    def build_schema(schema, title):
        schema = resolve(schema)
        properties = schema.get("properties") or {}
        if not isinstance(properties, dict):
            schema_error("properties must be an object")
        if schema.get("additionalProperties") not in (None, False):
            schema_error("additionalProperties is not supported")

        required = set(schema.get("required") or [])
        apple_properties = []
        nested_schemas = []
        nested_names = set()
        for name, property_schema in properties.items():
            value_type, property_nested = field_type(
                property_schema,
                f"{title}{_schema_title(name)}",
            )
            if name not in required:
                value_type = Optional[value_type]
            for nested_schema in property_nested:
                nested_name = nested_schema.type_class.__name__
                if nested_name not in nested_names:
                    nested_names.add(nested_name)
                    nested_schemas.append(nested_schema)
            apple_properties.append(
                Property(
                    name=name,
                    type_class=value_type,
                    description=property_schema.get("description"),
                    guides=guides_for(resolve(property_schema)),
                )
            )

        schema_type = type(title, (), {})
        return fm.GenerationSchema(
            type_class=schema_type,
            description=(
                schema.get("description")
                or tool.description
                or f"Arguments for {tool.name}"
            ),
            properties=apple_properties,
            dynamic_nested_types=nested_schemas,
        )

    return build_schema(
        root_schema,
        f"{_schema_title(tool.name)}Arguments",
    )


def _create_apple_tools(fm, tools, run_tool: Callable[[llm.ToolCall], Awaitable[str]]):
    apple_tools = []
    for tool in tools:
        arguments_schema = _tool_schema(fm, tool)

        async def call(self, args, llm_tool=tool):
            return await run_tool(
                llm.ToolCall(
                    name=llm_tool.name,
                    arguments=args.value(),
                )
            )

        def get_arguments_schema(self, schema=arguments_schema):
            return schema

        tool_class = type(
            f"{_schema_title(tool.name)}Tool",
            (fm.Tool,),
            {
                "name": tool.name,
                "description": tool.description or f"Call {tool.name}",
                "arguments_schema": property(get_arguments_schema),
                "call": call,
            },
        )
        apple_tools.append(tool_class())
    return apple_tools


def _tool_events(tool_call: llm.ToolCall, tool_result: llm.ToolResult):
    """StreamEvents recording a tool call that Apple's session ran mid-response.

    These are marked server_executed because the call and its result both
    happen inside this single response: LLM must not execute them again.
    """
    metadata = {PROVIDER: {"provider_managed": True}}
    common = {
        "tool_call_id": tool_result.tool_call_id,
        "server_executed": True,
        "provider_metadata": metadata,
    }
    return [
        StreamEvent(type="tool_call_name", chunk=tool_call.name, **common),
        StreamEvent(
            type="tool_call_args",
            chunk=json.dumps(tool_call.arguments or {}),
            **common,
        ),
        StreamEvent(
            type="tool_result",
            chunk=tool_result.output,
            tool_name=tool_call.name,
            **common,
        ),
    ]


def _new_id():
    return str(uuid.uuid4()).upper()


def _text_segments(texts):
    return [{"type": "text", "id": _new_id(), "text": text} for text in texts]


def _unsupported_attachments():
    return llm.ModelError("Apple Foundation models do not support attachments yet")


def _replayable(part):
    # Tool activity from another provider's server-side tools cannot be
    # replayed. Tool calls run by this plugin are also marked
    # server_executed, but carry our namespace in provider_metadata.
    return not part.server_executed or PROVIDER in (part.provider_metadata or {})


class _TranscriptBuilder:
    """Convert llm.Message history into Foundation Models transcript entries.

    The entry format matches what ``Transcript.to_dict()`` produces and what
    ``Transcript.from_dict()`` accepts. Every entry and segment needs an id,
    and prompt entries need an ``options`` dictionary.
    """

    def __init__(self):
        self.entries = []
        self._role = None
        self._texts = []
        self._tool_calls = []

    def add_message(self, message):
        for part in message.parts:
            if isinstance(part, TextPart):
                if part.text:
                    self._add_text(
                        "response" if message.role == "assistant" else "user",
                        part.text,
                    )
            elif isinstance(part, ToolCallPart):
                if _replayable(part):
                    self._add_tool_call(part)
            elif isinstance(part, ToolResultPart):
                if _replayable(part):
                    self._add_tool_output(part)
            elif isinstance(part, AttachmentPart):
                raise _unsupported_attachments()
            # ReasoningPart has no Foundation Models equivalent
        self._flush()

    def _add_text(self, role, text):
        if self._role != role:
            self._flush()
            self._role = role
        self._texts.append(text)

    def _add_tool_call(self, part):
        self._flush_texts()
        self._tool_calls.append(
            {
                "id": part.tool_call_id or _new_id(),
                "name": part.name,
                "arguments": json.dumps(part.arguments or {}),
            }
        )

    def _add_tool_output(self, part):
        if part.attachments:
            raise _unsupported_attachments()
        self._flush()
        tool_call_id = part.tool_call_id or _new_id()
        self.entries.append(
            {
                "role": "tool",
                "id": tool_call_id,
                "toolCallID": tool_call_id,
                "toolName": part.name,
                "contents": _text_segments([part.output or ""]),
            }
        )

    def _flush_texts(self):
        if self._texts:
            entry = {
                "role": self._role,
                "id": _new_id(),
                "contents": _text_segments(self._texts),
            }
            if self._role == "user":
                entry["options"] = {}
            self.entries.append(entry)
        self._texts = []
        self._role = None

    def _flush(self):
        self._flush_texts()
        if self._tool_calls:
            self.entries.append(
                {"role": "response", "id": _new_id(), "toolCalls": self._tool_calls}
            )
            self._tool_calls = []


@dataclass
class _Turn:
    instructions: Optional[str]
    history: list
    prompt: Union[str, list[str]]


def _split_messages(messages) -> _Turn:
    """Split prompt.messages into instructions, prior history and new prompt.

    prompt.messages is the complete input chain for this turn, including
    history from earlier turns, other models and explicit messages= lists.
    """
    instructions = []
    history = []
    for message in messages:
        if message.role == "system":
            for part in message.parts:
                if isinstance(part, TextPart):
                    if part.text:
                        instructions.append(part.text)
                elif isinstance(part, AttachmentPart):
                    raise _unsupported_attachments()
        else:
            history.append(message)

    prompt: Union[str, list[str]] = ""
    if history and history[-1].role == "user":
        current = history.pop()
        texts = []
        for part in current.parts:
            if isinstance(part, AttachmentPart):
                raise _unsupported_attachments()
            if isinstance(part, TextPart) and part.text:
                texts.append(part.text)
        if len(texts) == 1:
            prompt = texts[0]
        elif texts:
            prompt = texts

    return _Turn(
        instructions="\n\n".join(instructions) or None,
        history=history,
        prompt=prompt,
    )


async def _create_session(fm, system_model, turn, tools):
    """Create a session with the turn's instructions, tools and history.

    Returns the session and the number of transcript entries it starts with.
    """
    session = fm.LanguageModelSession(
        instructions=turn.instructions,
        model=system_model,
        tools=tools,
    )
    transcript = await session.transcript.to_dict()
    if not turn.history:
        return session, len(transcript["transcript"]["entries"])

    builder = _TranscriptBuilder()
    for message in turn.history:
        builder.add_message(message)
    # Keep the instructions entry Apple created, which describes the tools
    transcript["transcript"]["entries"].extend(builder.entries)
    restored = fm.LanguageModelSession.from_transcript(
        await fm.Transcript.from_dict(transcript),
        model=system_model,
        tools=tools,
    )
    return restored, len(transcript["transcript"]["entries"])


async def _execute_async(
    prompt, stream, response, execute_tool_call: ToolExecutor
) -> AsyncGenerator[Union[str, StreamEvent], None]:
    fm = _load_sdk()
    system_model = _create_system_model(fm, prompt.options)
    _record_model_variant(system_model, response)
    generation_options = _create_generation_options(fm, prompt.options)
    turn = _split_messages(prompt.messages)

    # apple-fm-sdk runs each tool on its own thread and event loop, and its
    # stream_response() blocks this event loop while waiting for snapshots.
    # Tools therefore execute where the SDK calls them, recording their
    # events in a thread-safe queue that is drained between snapshots.
    tool_events: queue.Queue = queue.Queue()

    async def run_tool(tool_call):
        tool_result = await execute_tool_call(tool_call)
        if tool_result.attachments:
            raise ValueError(
                "Apple Foundation tool output attachments are not supported yet"
            )
        for event in _tool_events(tool_call, tool_result):
            tool_events.put(event)
        return tool_result.output

    def recorded_tool_events():
        while True:
            try:
                yield tool_events.get_nowait()
            except queue.Empty:
                return

    apple_tools = _create_apple_tools(fm, prompt.tools, run_tool)
    try:
        session, existing_entries = await _create_session(
            fm, system_model, turn, apple_tools
        )
        if prompt.schema:
            result = await session.respond(
                turn.prompt,
                json_schema=_normalize_schema(prompt.schema),
                options=generation_options,
            )
            for event in recorded_tool_events():
                yield event
            yield result.to_json()
        elif stream:
            previous_snapshot = ""
            async for snapshot in session.stream_response(
                turn.prompt,
                options=generation_options,
            ):
                for event in recorded_tool_events():
                    yield event
                if not snapshot.startswith(previous_snapshot):
                    raise llm.ModelError(
                        "Apple Foundation model returned a non-monotonic stream"
                    )
                delta = snapshot[len(previous_snapshot) :]
                previous_snapshot = snapshot
                if delta:
                    yield delta
        else:
            result = await session.respond(
                turn.prompt,
                options=generation_options,
            )
            for event in recorded_tool_events():
                yield event
            yield result
        for event in recorded_tool_events():
            yield event
        transcript = await session.transcript.to_dict()
    except fm.ExceededContextWindowSizeError as ex:
        raise llm.ModelError(
            "This conversation exceeds the Apple Foundation model's context "
            f"window of {system_model.context_size} tokens. Start a new "
            "conversation or use a shorter prompt."
        ) from ex
    except fm.FoundationModelsError as ex:
        raise llm.ModelError(str(ex)) from ex

    # Record the transcript entries Apple added for this turn
    entries = transcript["transcript"]["entries"]
    transcript["transcript"]["entries"] = entries[existing_entries:]
    response.response_json = transcript


def _sync_iterate(
    async_generator_factory: Callable[
        [ToolExecutor], AsyncGenerator[Union[str, StreamEvent], None]
    ],
    response: llm.Response,
) -> Iterator[Union[str, StreamEvent]]:
    items = queue.Queue()
    finished = object()

    async def execute_tool_call(tool_call):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        items.put(_ToolExecutionRequest(tool_call, loop, future))
        return await future

    async def consume():
        try:
            async for item in async_generator_factory(execute_tool_call):
                items.put(item)
        except BaseException as ex:
            items.put(ex)
        finally:
            items.put(finished)

    thread = threading.Thread(target=lambda: asyncio.run(consume()), daemon=True)
    thread.start()

    while True:
        item = items.get()
        if item is finished:
            break
        if isinstance(item, _ToolExecutionRequest):
            try:
                tool_result = response.execute_tool_call(item.tool_call)
            except Exception as ex:
                item.loop.call_soon_threadsafe(item.future.set_exception, ex)
            else:
                item.loop.call_soon_threadsafe(item.future.set_result, tool_result)
            continue
        if isinstance(item, BaseException):
            raise item
        yield item

    thread.join()


class AppleFoundationModel(llm.Model):
    model_id = "apple-foundation"
    supports_schema = True
    supports_tools = True
    Options = AppleFoundationOptions

    def execute(self, prompt, stream, response, conversation=None):
        # Conversation history is read from prompt.messages, not conversation
        yield from _sync_iterate(
            lambda execute_tool_call: _execute_async(
                prompt,
                stream,
                response,
                execute_tool_call,
            ),
            response,
        )


class AsyncAppleFoundationModel(llm.AsyncModel):
    model_id = "apple-foundation"
    supports_schema = True
    supports_tools = True
    Options = AppleFoundationOptions

    async def execute(self, prompt, stream, response, conversation=None):
        async for chunk in _execute_async(
            prompt,
            stream,
            response,
            response.execute_tool_call,
        ):
            yield chunk


@llm.hookimpl
def register_models(register):
    register(
        AppleFoundationModel(),
        AsyncAppleFoundationModel(),
        aliases=("afm",),
    )
