"""LLM backends behind one tool-calling chat interface.

- OpenAICompatClient: any OpenAI-compatible server. This is the single path
  for BOTH cloud OpenAI-style APIs and the local DGX Spark stack (llama.cpp
  llama-server / vLLM serving Qwen3.6 with MTP speculative decoding expose
  the same /v1/chat/completions API).
- AnthropicClient: Claude over the Anthropic API (tool use blocks).
- MockLLM: scripted responses for tests and --llm mock dry-runs.

Images ride along as base64 JPEG (data URLs for OpenAI-compat, source blocks
for Anthropic); backends that are text-only simply ignore them.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..config import Cfg


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # vLLM with --reasoning-parser qwen3 populates this field with the
    # content inside <think>...</think>. Cloud / non-reasoning models
    # leave it empty. Reasoning models that emit tool calls inside the
    # think block fall back to this when content is null.
    reasoning: str = ""


class LLMConfigError(RuntimeError):
    """Raised by make_llm() when a profile is misconfigured (missing env var,
    unknown type, etc.). Surfaces at startup so the operator sees the env var
    name in the error message, not at chat time. See ADR-0008."""

    def __init__(self, profile_name: str | None, env_key: str | None, *, hint: str = ""):
        self.profile_name = profile_name
        self.env_key = env_key
        msg_parts = []
        if profile_name:
            msg_parts.append(f"profile {profile_name!r}")
        if env_key:
            msg_parts.append(f"env var {env_key!r}")
        detail = "; ".join(msg_parts) or "LLM client construction"
        msg = f"{detail}: missing or invalid configuration"
        if hint:
            msg += f" — {hint}"
        super().__init__(msg)


class LLMClient:
    supports_vision = False

    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """messages: [{role, content:str, images?: [jpeg bytes]}, ...] where a
        role="tool" message carries {tool_call_id, name, content}."""
        raise NotImplementedError


def _b64(jpeg: bytes) -> str:
    return base64.b64encode(jpeg).decode()


def _parse_hermes_tool_calls(content: str) -> list[ToolCall]:
    """Recover Hermes tool calls emitted in message content.

    Some OpenAI-compatible servers return ``<tool_call>{...}</tool_call>``
    as plain text even when their Hermes parser is enabled. JSONDecoder is
    used instead of a regex so nested argument objects parse correctly.
    """
    calls: list[ToolCall] = []
    marker = "<tool_call>"
    decoder = json.JSONDecoder()
    cursor = 0
    while True:
        marker_at = content.find(marker, cursor)
        if marker_at < 0:
            break
        payload_at = marker_at + len(marker)
        while payload_at < len(content) and content[payload_at].isspace():
            payload_at += 1
        try:
            payload, consumed = decoder.raw_decode(content, payload_at)
        except json.JSONDecodeError:
            cursor = payload_at
            continue
        cursor = consumed
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            continue
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
        if not isinstance(arguments, dict):
            arguments = {"_raw": arguments}
        calls.append(
            ToolCall(
                name=payload["name"],
                arguments=arguments,
                id=str(payload.get("id", "")),
            )
        )
    return calls


class OpenAICompatClient(LLMClient):
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        supports_vision: bool = True,
        temperature: float = 0.2,
    ):
        import os

        import httpx
        from openai import OpenAI

        # Local servers (llama.cpp/vLLM) need a dummy key; hosted endpoints
        # fall through to the OPENAI_API_KEY environment variable.
        if api_key is None and base_url is not None and "OPENAI_API_KEY" not in os.environ:
            api_key = "EMPTY"
        # Bypass the host's HTTP/SOCKS proxy env vars -- some demo rigs
        # have ALL_PROXY=socks://... set globally, and httpx (the OpenAI
        # SDK's HTTP transport) doesn't support SOCKS proxies, so the
        # client crashes at construction time. `trust_env=False` on the
        # httpx client disables env-var-driven proxy detection. The
        # endpoint URL itself is already explicit via base_url, so we
        # never want a proxy for direct-API calls.
        proxy_bypass_http = httpx.Client(trust_env=False)
        self._client = OpenAI(
            base_url=base_url, api_key=api_key, http_client=proxy_bypass_http
        )
        self.model = model
        self.supports_vision = supports_vision
        self.temperature = temperature

    def chat(self, system, messages, tools=None, max_tokens=1024) -> LLMResponse:
        oai_msgs: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "tool":
                oai_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.get("tool_call_id", "call_0"),
                        "content": m["content"],
                    }
                )
                continue
            if m["role"] == "assistant" and m.get("tool_calls"):
                oai_msgs.append(
                    {
                        "role": "assistant",
                        "content": m.get("content") or None,
                        "tool_calls": [
                            {
                                "id": tc.id or f"call_{i}",
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for i, tc in enumerate(m["tool_calls"])
                        ],
                    }
                )
                continue
            images = m.get("images") or []
            if images and self.supports_vision:
                content: Any = [{"type": "text", "text": m["content"]}]
                for jpeg in images:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{_b64(jpeg)}"},
                        }
                    )
            else:
                content = m["content"]
            oai_msgs.append({"role": m["role"], "content": content})

        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=oai_msgs,
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": t} for t in tools
            ]
            # One tool call per turn keeps the loop deterministic.
            kwargs["parallel_tool_calls"] = False
        resp = self._call_with_param_fallback(kwargs)
        choice = resp.choices[0].message
        calls = []
        for tc in choice.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(name=tc.function.name, arguments=args, id=tc.id))
        if not calls and choice.content:
            calls = _parse_hermes_tool_calls(choice.content)
        # choice.reasoning is set by vLLM with --reasoning-parser qwen3;
        # getattr guards against the OpenAI SDK Message class not exposing
        # the attribute on cloud/non-reasoning models.
        reasoning = getattr(choice, "reasoning", None) or ""
        return LLMResponse(text=choice.content or "", reasoning=reasoning, tool_calls=calls)

    def _call_with_param_fallback(self, kwargs: dict):
        """Newer OpenAI models reject max_tokens (want max_completion_tokens);
        some local servers reject parallel_tool_calls. A local model may also
        report its exact context/input token counts. In that case, shrink only
        the output allowance (with a small safety margin) and retry instead of
        shutting down the robot application before a tool can run."""
        for _ in range(5):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as e:
                msg = str(e)
                if "max_completion_tokens" in msg and "max_tokens" in kwargs:
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                    continue
                if "parallel_tool_calls" in msg and "parallel_tool_calls" in kwargs:
                    kwargs.pop("parallel_tool_calls")
                    continue
                if "temperature" in msg and "temperature" in kwargs:
                    kwargs.pop("temperature")
                    continue
                match = re.search(
                    r"maximum context length is\s+(\d+)\s+tokens.*?"
                    r"prompt contains at least\s+(\d+)\s+input tokens",
                    msg,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                token_key = (
                    "max_tokens" if "max_tokens" in kwargs
                    else "max_completion_tokens"
                    if "max_completion_tokens" in kwargs else None
                )
                if match and token_key is not None:
                    context_tokens, input_tokens = map(int, match.groups())
                    # The server says "at least" for the prompt size. Keep 32
                    # tokens of uncertainty rather than retrying at the exact
                    # boundary (the failure motivating this was over by 1).
                    safe_output = context_tokens - input_tokens - 32
                    current_output = int(kwargs[token_key])
                    if 16 <= safe_output < current_output:
                        kwargs[token_key] = safe_output
                        continue
                raise
        return self._client.chat.completions.create(**kwargs)


class AnthropicClient(LLMClient):
    supports_vision = True

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        temperature: float = 0.2,
        base_url: str | None = None,
        api_key: str | None = None,
        auth_token: str | None = None,
        supports_vision: bool = True,
    ):
        import anthropic

        client_kwargs: dict[str, Any] = {}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if auth_token is not None:
            client_kwargs["auth_token"] = auth_token
        self._client = anthropic.Anthropic(**client_kwargs)
        self.model = model
        self.temperature = temperature
        self.supports_vision = supports_vision

    def chat(self, system, messages, tools=None, max_tokens=1024) -> LLMResponse:
        anth_msgs: list[dict] = []
        for m in messages:
            if m["role"] == "tool":
                anth_msgs.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.get("tool_call_id", "call_0"),
                                "content": m["content"],
                            }
                        ],
                    }
                )
                continue
            if m["role"] == "assistant" and m.get("tool_calls"):
                blocks: list[dict] = []
                if (m.get("content") or "").strip():
                    blocks.append({"type": "text", "text": m["content"]})
                for i, tc in enumerate(m["tool_calls"]):
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id or f"call_{i}",
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                anth_msgs.append({"role": "assistant", "content": blocks})
                continue
            # The API rejects empty text blocks; substitute a placeholder.
            text = m["content"] if (m.get("content") or "").strip() else "(no text)"
            content: list[dict] = [{"type": "text", "text": text}]
            for jpeg in m.get("images") or []:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": _b64(jpeg),
                        },
                    }
                )
            anth_msgs.append({"role": m["role"], "content": content})

        kwargs: dict[str, Any] = dict(
            model=self.model,
            system=system,
            messages=anth_msgs,
            max_tokens=max_tokens,
        )
        # anthropic SDK 1.3.0+ removed `temperature` from the typed signature;
        # route via `extra_body` (universal pass-through) to keep this
        # client working without pinning an older SDK.
        kwargs["extra_body"] = {"temperature": self.temperature}
        if tools:
            kwargs["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object"}),
                }
                for t in tools
            ]
            kwargs["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        resp = self._client.messages.create(**kwargs)
        text_parts, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(name=block.name, arguments=dict(block.input), id=block.id))
        return LLMResponse(text="\n".join(text_parts), tool_calls=calls)


class MockLLM(LLMClient):
    """Pops scripted responses; records every request for assertions."""

    supports_vision = True

    def __init__(self, script: list[LLMResponse]):
        self._script = list(script)
        self.requests: list[dict] = []

    def chat(self, system, messages, tools=None, max_tokens=1024) -> LLMResponse:
        # Snapshot the (mutable, reused) message list so per-request
        # assertions in tests see what was actually sent at that time.
        self.requests.append(
            {"system": system, "messages": [dict(m) for m in messages], "tools": tools}
        )
        if not self._script:
            return LLMResponse(text="(mock script exhausted)", tool_calls=[])
        return self._script.pop(0)


def make_llm(cfg: Cfg) -> LLMClient:
    kind = cfg.type
    if kind == "mock":
        # Default wiring-check script: observe once, then finish honestly.
        return MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall("get_observation", {})]),
                LLMResponse(
                    tool_calls=[
                        ToolCall(
                            "task_done",
                            {
                                "success": True,
                                "summary": "mock LLM wiring check: observed the scene and stopped",
                            },
                        )
                    ]
                ),
            ]
        )
    if kind == "anthropic":
        import os as _os

        _auth_token_env = cfg.get("auth_token_env")
        _auth_token = cfg.get("auth_token")
        if _auth_token is None and _auth_token_env:
            _auth_token = _os.environ.get(_auth_token_env)
        if _auth_token_env and not _auth_token:
            raise LLMConfigError(
                profile_name=cfg.get("profile") or cfg.get("name"),
                env_key=_auth_token_env,
                hint=f"export {_auth_token_env}=<your-token> before launching",
            )

        _api_key_env = cfg.get("api_key_env")
        _api_key = cfg.get("api_key")
        if _api_key is None and _api_key_env:
            _api_key = _os.environ.get(_api_key_env)
        if _api_key_env and not _api_key:
            raise LLMConfigError(
                profile_name=cfg.get("profile") or cfg.get("name"),
                env_key=_api_key_env,
                hint=f"export {_api_key_env}=<your-key> before launching",
            )

        return AnthropicClient(
            model=cfg.get("model", "claude-sonnet-5"),
            temperature=float(cfg.get("temperature", 0.2)),
            base_url=cfg.get("base_url"),
            api_key=_api_key,
            auth_token=_auth_token,
            supports_vision=bool(cfg.get("supports_vision", True)),
        )
    if kind == "openai_compat":
        # Read api_key from yaml OR from the env var named by `env_key`
        # (defaults to OPENAI_API_KEY). Avoids putting secrets in yaml.
        import os as _os
        _api_key = cfg.get("api_key")
        _env_var = cfg.get("env_key")  # may be None — treat as "no key required" (local dummy)
        if _api_key is None and _env_var is not None:
            _api_key = _os.environ.get(_env_var)
        elif _api_key is None and _env_var is None:
            # Default to OPENAI_API_KEY for backward compatibility with profiles
            # that pre-date ADR-0006 (no env_key declared).
            _api_key = _os.environ.get("OPENAI_API_KEY")
        if _env_var and not _api_key:
            # ADR-0006: profile declares env_key but env is unset → misconfig,
            # surface at startup not at chat time.
            raise LLMConfigError(
                profile_name=cfg.get("profile") or cfg.get("name"),
                env_key=_env_var,
                hint=f"export {_env_var}=<your-key> before launching",
            )
        return OpenAICompatClient(
            model=cfg.model,
            base_url=cfg.get("base_url"),
            api_key=_api_key or "EMPTY",
            supports_vision=bool(cfg.get("supports_vision", True)),
            temperature=float(cfg.get("temperature", 0.2)),
        )
    raise ValueError(f"unknown llm type {kind!r} (mock|anthropic|openai_compat)")
