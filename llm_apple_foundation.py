import asyncio
import base64
import copy
import inspect
import json
import queue
import threading
import zlib
from typing import (
    AsyncGenerator,
    Callable,
    Iterator,
    Literal,
    Optional,
)

import llm
from pydantic import Field, model_validator


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


async def _call_llm_tool(tool, arguments):
    if not tool.implementation:
        raise ValueError(f"No implementation available for tool: {tool.name}")

    tool_call = llm.ToolCall(name=tool.name, arguments=arguments)
    implementation_arguments = dict(arguments)
    try:
        signature = inspect.signature(tool.implementation)
    except (TypeError, ValueError):
        signature = None
    if signature and "llm_tool_call" in signature.parameters:
        implementation_arguments["llm_tool_call"] = tool_call

    result = tool.implementation(**implementation_arguments)
    if inspect.isawaitable(result):
        result = await result
    if isinstance(result, llm.ToolOutput):
        if result.attachments:
            raise ValueError(
                "Apple Foundation tool output attachments are not supported yet"
            )
        result = result.output
    if isinstance(result, str):
        return result
    return json.dumps(result, default=repr)


def _create_apple_tools(fm, tools):
    apple_tools = []
    for tool in tools:
        arguments_schema = _tool_schema(fm, tool)

        async def call(self, args, llm_tool=tool):
            return await _call_llm_tool(llm_tool, args.value())

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


_TRANSCRIPT_ENCODING = "base64+zlib+json"


def _pack_transcript(transcript):
    raw = json.dumps(
        transcript,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "type": "FoundationModels.Transcript",
        "encoding": _TRANSCRIPT_ENCODING,
        "data": base64.b64encode(zlib.compress(raw)).decode("ascii"),
    }


def _unpack_transcript(value):
    if (
        not isinstance(value, dict)
        or value.get("type") != "FoundationModels.Transcript"
    ):
        return None
    if value.get("encoding") == _TRANSCRIPT_ENCODING:
        try:
            compressed = base64.b64decode(value["data"])
            return json.loads(zlib.decompress(compressed).decode("utf-8"))
        except (KeyError, TypeError, ValueError, zlib.error) as ex:
            raise llm.ModelError(
                "Stored Apple Foundation transcript is invalid"
            ) from ex
    # Support transcripts logged by earlier development versions.
    if "transcript" in value:
        return value
    return None


async def _create_session(fm, system_model, prompt, conversation, tools):
    if conversation:
        for previous_response in reversed(conversation.responses):
            transcript_data = _unpack_transcript(previous_response.response_json)
            if transcript_data:
                transcript = await fm.Transcript.from_dict(transcript_data)
                return fm.LanguageModelSession.from_transcript(
                    transcript,
                    model=system_model,
                    tools=tools,
                )

    return fm.LanguageModelSession(
        instructions=prompt.system,
        model=system_model,
        tools=tools,
    )


async def _execute_async(
    prompt, stream, response, conversation
) -> AsyncGenerator[str, None]:
    fm = _load_sdk()
    system_model = _create_system_model(fm, prompt.options)
    generation_options = _create_generation_options(fm, prompt.options)
    apple_tools = _create_apple_tools(fm, prompt.tools)
    session = await _create_session(
        fm,
        system_model,
        prompt,
        conversation,
        apple_tools,
    )
    try:
        if prompt.schema:
            result = await session.respond(
                prompt.prompt or "",
                json_schema=_normalize_schema(prompt.schema),
                options=generation_options,
            )
            yield result.to_json()
        elif stream:
            previous_snapshot = ""
            async for snapshot in session.stream_response(
                prompt.prompt or "",
                options=generation_options,
            ):
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
                prompt.prompt or "",
                options=generation_options,
            )
            yield result
        response.response_json = _pack_transcript(await session.transcript.to_dict())
    except fm.FoundationModelsError as ex:
        raise llm.ModelError(str(ex)) from ex


def _sync_iterate(
    async_generator_factory: Callable[[], AsyncGenerator[str, None]],
) -> Iterator[str]:
    items = queue.Queue()
    finished = object()

    async def consume():
        try:
            async for item in async_generator_factory():
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
        yield from _sync_iterate(
            lambda: _execute_async(prompt, stream, response, conversation)
        )


class AsyncAppleFoundationModel(llm.AsyncModel):
    model_id = "apple-foundation"
    supports_schema = True
    supports_tools = True
    Options = AppleFoundationOptions

    async def execute(self, prompt, stream, response, conversation=None):
        async for chunk in _execute_async(prompt, stream, response, conversation):
            yield chunk


@llm.hookimpl
def register_models(register):
    register(
        AppleFoundationModel(),
        AsyncAppleFoundationModel(),
        aliases=("afm",),
    )
