import asyncio
import copy
import queue
import threading
from typing import AsyncGenerator, Callable, Iterator, Literal, Optional

import llm
from pydantic import Field


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


async def _create_session(fm, system_model, prompt, conversation):
    if conversation:
        for previous_response in reversed(conversation.responses):
            transcript_data = previous_response.response_json
            if (
                isinstance(transcript_data, dict)
                and transcript_data.get("type") == "FoundationModels.Transcript"
            ):
                transcript = await fm.Transcript.from_dict(transcript_data)
                return fm.LanguageModelSession.from_transcript(
                    transcript,
                    model=system_model,
                )

    return fm.LanguageModelSession(
        instructions=prompt.system,
        model=system_model,
    )


async def _execute_async(
    prompt, stream, response, conversation
) -> AsyncGenerator[str, None]:
    fm = _load_sdk()
    system_model = _create_system_model(fm, prompt.options)
    generation_options = _create_generation_options(fm, prompt.options)
    session = await _create_session(
        fm,
        system_model,
        prompt,
        conversation,
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
        response.response_json = await session.transcript.to_dict()
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
    Options = AppleFoundationOptions

    def execute(self, prompt, stream, response, conversation=None):
        yield from _sync_iterate(
            lambda: _execute_async(prompt, stream, response, conversation)
        )


class AsyncAppleFoundationModel(llm.AsyncModel):
    model_id = "apple-foundation"
    supports_schema = True
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
