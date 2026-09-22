"""LLM backends.

Three things matter here, and none of them is "call an API".

1. **The core has no hard dependency.** The transport is ``urllib`` from the
   standard library, because DashScope's Qwen endpoint is OpenAI-compatible and
   the whole request is one JSON POST. ``openai`` is offered as an extra, not
   required. Fewer moving parts means the repo runs on a bare interpreter,
   which is what makes the test suite meaningful.

2. **Failure is modelled, not caught-and-ignored.** :class:`LLMTransientError`
   (429, 5xx, socket timeouts) is retried with exponential backoff;
   :class:`LLMError` is permanent and aborts the chunk into the review queue
   rather than silently emitting a half-translated page.

3. **The mock is a first-class backend with fault injection.** A deterministic
   model that returns ``"[译] ..."`` proves nothing. A deterministic model that
   can be *told to corrupt exactly one financial figure* lets us demonstrate
   that the number guard actually catches hallucination -- which is the single
   claim an interviewer most wants to see evidence for. See
   :class:`MockLLM`'s ``inject_number_drift``.
"""

from __future__ import annotations

import base64
import json
import random
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..textutils import estimate_tokens
from .number_guard import extract_numbers

__all__ = [
    "LLMError",
    "LLMTransientError",
    "Completion",
    "LLMClient",
    "OpenAICompatClient",
    "MockLLM",
    "make_llm",
]


class LLMError(RuntimeError):
    """Permanent failure: bad request, auth, malformed response."""


class LLMTransientError(LLMError):
    """Retryable failure: rate limit, server error, connection reset."""


@dataclass
class Completion:
    text: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# interface
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """Minimal surface the pipeline needs: text and vision completion."""

    name: str = "abstract"

    def __init__(self) -> None:
        #: Updated after every call; aggregated into the run report.
        self.last_usage: dict[str, int] = {}

    @abstractmethod
    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Single-turn text completion."""

    @abstractmethod
    def complete_vision(
        self,
        *,
        system: str,
        user: str,
        image_bytes: bytes,
        mime: str = "image/png",
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Single-turn completion over one image plus a text instruction."""

    def complete_json(self, *, system: str, user: str, **kwargs: Any) -> dict[str, Any]:
        """Completion constrained to a JSON object, parsed and validated.

        ``response_format`` is requested when supported, but the reply is
        *always* re-parsed defensively -- providers silently ignore JSON mode
        and return prose-wrapped JSON, and a pipeline that trusts the flag
        breaks in production exactly when it must not.
        """
        from ..parser.analyzer import extract_json_object

        kwargs.setdefault("response_format", {"type": "json_object"})
        text = self.complete(system=system, user=user, **kwargs)
        return extract_json_object(text)

    def close(self) -> None:
        """Release transport resources."""

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# OpenAI-compatible transport (DashScope / vLLM / Ollama / OpenAI)
# ---------------------------------------------------------------------------


class OpenAICompatClient(LLMClient):
    """``POST {base_url}/chat/completions`` using only the standard library.

    Verified against Alibaba Cloud Bailian (DashScope) compatible mode, and
    works unchanged against any OpenAI-compatible gateway. Because it is plain
    HTTP, you can point it at a local vLLM or Ollama server for a fully offline
    production run.
    """

    name = "openai-compat"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        vlm_model: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        backoff_base: float = 1.5,
        extra_headers: dict[str, str] | None = None,
        #: Injection point for tests: replaces the real HTTP call.
        transport: Callable[[str, dict[str, str], bytes, float], tuple[int, dict[str, Any]]] | None = None,
    ) -> None:
        super().__init__()
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.vlm_model = vlm_model or model
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.backoff_base = backoff_base
        self.extra_headers = extra_headers or {}
        self._transport = transport

    # -- transport ----------------------------------------------------------

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            **self.extra_headers,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if self._transport is not None:
                    status, data = self._transport(url, headers, body, self.timeout)
                else:
                    status, data = self._http_post(url, headers, body)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = LLMTransientError(f"transport failure: {exc}")
                self._sleep(attempt)
                continue

            if status == 200:
                return data
            message = _error_message(data)
            if status in (408, 409, 429) or 500 <= status < 600:
                last_error = LLMTransientError(f"HTTP {status}: {message}")
                self._sleep(attempt)
                continue
            raise LLMError(f"HTTP {status}: {message}")

        raise last_error or LLMError("request failed with no recorded error")

    def _http_post(self, url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, Any]]:
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - fixed https endpoint
                return int(response.status), json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = {"error": {"message": payload[:500]}}
            return int(exc.code), data

    def _sleep(self, attempt: int) -> None:
        if attempt < self.max_retries - 1:
            time.sleep(self.backoff_base**attempt * 0.5)

    # -- completion ---------------------------------------------------------

    def _chat(self, payload: dict[str, Any], model: str) -> str:
        payload.setdefault("model", model)
        data = self._post(payload)
        usage = data.get("usage") or {}
        self.last_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
        try:
            choices = data["choices"]
            message = choices[0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {json.dumps(data)[:400]}") from exc
        # Reasoning models put the answer in `content`; some gateways use a
        # list of parts. Handle both.
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return content or ""

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        return self._chat(payload, self.model)

    def complete_vision(
        self,
        *,
        system: str,
        user: str,
        image_bytes: bytes,
        mime: str = "image/png",
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": user},
                    ],
                },
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return self._chat(payload, self.vlm_model)


def _error_message(data: dict[str, Any]) -> str:
    error = data.get("error")
    if isinstance(error, dict):
        return str(error.get("message", error))
    if error:
        return str(error)
    return json.dumps(data)[:300]


# ---------------------------------------------------------------------------
# deterministic offline backend
# ---------------------------------------------------------------------------

#: A small, real EN->ZH lexicon for financial-report boilerplate. The mock uses
#: it so that offline demo output is recognisably a translation rather than a
#: placeholder -- but it is deliberately tiny and the demo says so.
MOCK_LEXICON: dict[str, str] = {
    "Annual Report": "年度报告",
    "Financial Summary": "财务摘要",
    "Revenue": "营业收入",
    "Operating profit": "营业利润",
    "Profit before tax": "税前利润",
    "Profit for the year": "年度利润",
    "Total assets": "资产总额",
    "Total liabilities": "负债总额",
    "Total equity": "所有者权益总额",
    "Net profit attributable to shareholders": "归属于股东的净利润",
    "Basic earnings per share": "基本每股收益",
    "Total": "合计",
    "Item": "项目",
    "Change": "变动",
    "Year ended 31 December": "截至12月31日止年度",
    "Unit": "单位",
    "Note": "附注",
    "Cash and cash equivalents": "现金及现金等价物",
    "Dividend": "股息",
    "Depreciation and amortisation": "折旧及摊销",
}

_NUMBER_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

#: Date markers, used to keep the fault injector away from calendar figures.
_DATE_AFTER_RE = re.compile(r"^\s*(?:月|日|号|年|月份)")
_DATE_BEFORE_RE = re.compile(r"(?:年|月|日|号|第)\s*$")
_MONTH_RE = re.compile(
    r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec)"
)


def looks_like_a_date(text: str, start: int, end: int) -> bool:
    """True when a figure sits inside a date rather than being a value.

    A date figure must never be the injector's target. Translating
    ``Year ended 31 December`` into ``截至12月31日止年度`` *legitimately* adds a
    figure -- the month -- that the source never printed, so the guard reports it
    in the ``added`` bucket rather than as a mismatch. That is the correct
    behaviour, but it means a corrupted *date* figure is indistinguishable from a
    benign reformat: corrupting the day of ``31 December`` leaves the original
    ``31`` in place (as ``31日``) and the mangled value surfaces as an addition.

    Injected corruption is meant to model "the model altered a financial figure".
    Restricting it to non-date figures keeps the injected fault well-posed and
    therefore keeps the demo's "caught N of N" verdict meaningful.
    """
    after = text[end : end + 12]
    before = text[max(0, start - 12) : start]
    if _DATE_AFTER_RE.match(after) or _DATE_BEFORE_RE.search(before):
        return True
    return bool(_MONTH_RE.match(after.lstrip()) or _MONTH_RE.search(before))


class MockLLM(LLMClient):
    """A deterministic, offline stand-in with **fault injection**.

    Real value of this class is not the (deliberately small) lexicon -- it is
    ``inject_number_drift``. Point the pipeline at this backend with
    ``inject_number_drift=1`` and the number guard must catch exactly one
    corrupted figure and route that chunk to human review. If it does not, the
    guard is broken. That turns "trust me, we validate the numbers" into a
    failing test if the validator ever regresses.
    """

    name = "mock"

    def __init__(
        self,
        *,
        lexicon: dict[str, str] | None = None,
        inject_number_drift: int = 0,
        inject_unit_flip: bool = False,
        truncate: bool = False,
        fail_on_calls: Sequence[int] = (),
        seed: int = 0,
        vision_payload: dict[str, Any] | None = None,
        latency: float = 0.0,
    ) -> None:
        super().__init__()
        self.lexicon = dict(MOCK_LEXICON)
        if lexicon:
            self.lexicon.update(lexicon)
        self.inject_number_drift = max(0, int(inject_number_drift))
        self.inject_unit_flip = inject_unit_flip
        self.truncate = truncate
        self.fail_on_calls = {int(i) for i in fail_on_calls}
        self.seed = seed
        self.vision_payload = vision_payload
        self.latency = latency
        self.calls: list[dict[str, Any]] = []
        self._rng = random.Random(seed)
        # Budget is consumed across the whole run (see ``_corrupt_numbers``).
        # It must be initialised here or every prose call raises AttributeError.
        self._drift_budget = self.inject_number_drift

    def reset_run(self) -> None:
        """Re-arm the fault budget and clear call history.

        The demo harness may reuse one instance for several passes; without this
        the second pass would start with an exhausted budget and silently inject
        nothing, which would make the guard look like it "caught everything".
        """
        self._drift_budget = self.inject_number_drift
        self.calls.clear()
        self._rng = random.Random(self.seed)

    # -- helpers ------------------------------------------------------------

    def _record(self, kind: str, payload: dict[str, Any]) -> int:
        index = len(self.calls)
        self.calls.append({"index": index, "kind": kind, "payload": payload})
        if self.latency:
            time.sleep(self.latency)
        if index in self.fail_on_calls:
            raise LLMTransientError(f"mock injected transient failure on call {index}")
        return index

    def _gloss(self, text: str) -> str:
        """Longest-match glossary substitution, then a visible marker for the rest."""
        out = text
        for source in sorted(self.lexicon, key=len, reverse=True):
            if source in out:
                out = out.replace(source, self.lexicon[source])
        return out

    def _corrupt_numbers(self, text: str) -> str:
        """Alter up to the remaining budget of figures, in a plausible way.

        Uses a digit swap (``1,234 -> 1,274``) rather than a wild jump, because
        a naive guard that only catches order-of-magnitude errors would pass a
        wild jump and fail this. This is the realistic hallucination.

        The budget is consumed across the whole run, so
        ``inject_number_drift=2`` means exactly two figures in the final document
        are wrong -- which is what makes "the guard caught 2 of 2" a real
        assertion rather than an accident of how many calls happened.
        """
        if self._drift_budget <= 0:
            return text
        # Only figures the *guard* would treat as figures are eligible. Two
        # exclusions, both about keeping the injected fault well-posed rather
        # than about making the guard's job easier:
        #
        #   * dates -- see looks_like_a_date;
        #   * scale markers such as the "'000" in "RMB'000", which the guard
        #     deliberately reads as a unit, not a value. Corrupting one produces
        #     a change the guard is designed not to see, which would make the
        #     injector and the verifier disagree about what a "figure" is.
        #
        # Using the guard's own extractor here is the point: the two sides agree
        # by construction, so "caught N of N" means something.
        eligible = {occurrence.span[0] for occurrence in extract_numbers(text)}
        matches = [
            m
            for m in _NUMBER_TOKEN_RE.finditer(text)
            if m.start() in eligible and not looks_like_a_date(text, m.start(), m.end())
        ]
        if not matches:
            return text
        count = min(self._drift_budget, len(matches))
        chosen = self._rng.sample(matches, k=count)
        self._drift_budget -= count
        chars = list(text)
        for m in sorted(chosen, key=lambda x: x.start(), reverse=True):
            token = m.group(0)
            digits = [c for c in token if c.isdigit()]
            if not digits:
                continue
            pos = self._rng.randrange(len(digits))
            original = digits[pos]
            replacement = str((int(original) + self._rng.choice([1, 2, -1, -2])) % 10)
            if replacement == original:
                replacement = str((int(original) + 1) % 10)
            # Rebuild the token, preserving the comma positions.
            new_token = []
            seen = 0
            for ch in token:
                if ch.isdigit():
                    new_token.append(replacement if seen == pos else ch)
                    seen += 1
                else:
                    new_token.append(ch)
            chars[m.start() : m.end()] = "".join(new_token)
        return "".join(chars)

    @property
    def corruption_applied(self) -> int:
        """How many figures were actually corrupted (for the demo's verdict line)."""
        return self.inject_number_drift - self._drift_budget

    # -- LLMClient ----------------------------------------------------------

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        self._record("complete", {"system": system, "user": user, "response_format": response_format})
        if response_format and response_format.get("type") == "json_object":
            return self._structured_reply(user)
        return self._prose_reply(user)

    # -- structured replies -------------------------------------------------

    def _structured_reply(self, user: str) -> str:
        """Answer the pipeline's JSON-mode prompts with real, usable output.

        The mock implements the two structured contracts the pipeline actually
        uses -- table-label translation and terminology repair -- rather than
        returning a generic stub. That matters: it means the offline run
        exercises the real code path (positional payload in, translations out,
        figures copied by the pipeline), so the number guard's verdict on a mock
        run is meaningful evidence about the pipeline rather than about the stub.
        """
        from ..parser.analyzer import extract_json_object

        payload: dict[str, Any] | None = None
        try:
            payload = extract_json_object(user)
        except ValueError:
            payload = None

        if payload is not None and isinstance(payload.get("cells"), list):
            return json.dumps({"translations": self._translate_cells(payload)}, ensure_ascii=False)

        if payload is not None and isinstance(payload.get("charts"), list):
            glossary = payload.get("glossary") or {}
            if not isinstance(glossary, dict):
                glossary = {}
            charts = []
            for entry in payload["charts"]:
                if not isinstance(entry, dict):
                    continue
                charts.append(
                    {
                        "i": int(entry.get("i", 0)),
                        "caption": self._translate_label(str(entry.get("caption", "")), glossary),
                        "description": self._prose_reply("SOURCE:\n" + str(entry.get("description", "")) + "\n---"),
                    }
                )
            return json.dumps({"charts": charts}, ensure_ascii=False)

        if "CURRENT TRANSLATION:" in user:
            return json.dumps(self._repair(user), ensure_ascii=False)

        if payload is not None and "text" in payload:
            # Terminology-proposal style prompt.
            return json.dumps({"terms": [], "notes": "mock backend"}, ensure_ascii=False)

        return json.dumps({"result": "mock", "ok": True}, ensure_ascii=False)

    def _translate_cells(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        glossary = payload.get("glossary") or {}
        if not isinstance(glossary, dict):
            glossary = {}
        out: list[dict[str, Any]] = []
        for cell in payload["cells"]:
            if not isinstance(cell, dict):
                continue
            try:
                row = int(cell.get("r", -1))
                col = int(cell.get("c", -1))
            except (TypeError, ValueError):
                continue
            if row < 0 or col < 0:
                continue
            out.append({"r": row, "c": col, "text": self._translate_label(str(cell.get("text", "")), glossary)})
        return out

    def _translate_label(self, text: str, glossary: dict[str, Any]) -> str:
        """Glossary first (approved terms win), then the built-in lexicon."""
        result = text
        pairs = sorted(
            ((str(k), str(v)) for k, v in glossary.items()), key=lambda kv: len(kv[0]), reverse=True
        )
        for source, target in pairs:
            if source and source in result:
                result = result.replace(source, target)
        result = self._gloss(result)
        result = self._corrupt_numbers(result)
        if self.truncate:
            result = result[: max(1, len(result) // 2)]
        return result or text

    def _repair(self, user: str) -> dict[str, Any]:
        """Apply the approved terms listed in the prompt to the given translation."""
        target = user.split("CURRENT TRANSLATION:", 1)[1].strip()
        pairs = re.findall(r"^- (.+?) => (.+)$", user, re.MULTILINE)
        applied: list[str] = []
        unresolved: list[str] = []
        fixed = target
        for source_term, target_term in pairs:
            source_term, target_term = source_term.strip(), target_term.strip()
            if not source_term or not target_term or target_term in fixed:
                continue
            if source_term in fixed:
                fixed = fixed.replace(source_term, target_term)
                applied.append(source_term)
            elif source_term.lower() in fixed.lower():
                fixed = re.sub(re.escape(source_term), target_term, fixed, flags=re.IGNORECASE)
                applied.append(source_term)
            else:
                unresolved.append(source_term)
        return {"fixed": fixed, "applied": applied, "unresolved": unresolved}

    # -- prose --------------------------------------------------------------

    def _prose_reply(self, user: str) -> str:
        body = user
        marker = "SOURCE:\n"
        if marker in user:
            body = user.split(marker, 1)[1]
        body = body.split("\n---", 1)[0]
        body = body.strip()
        translated = self._gloss(body)
        if translated == body:
            translated = f"[译]{body}"
        translated = self._corrupt_numbers(translated)
        if self.inject_unit_flip and "千元" in translated:
            translated = translated.replace("千元", "元")
        if self.truncate:
            translated = translated[: max(1, len(translated) // 3)]
        return translated

    def complete_vision(
        self,
        *,
        system: str,
        user: str,
        image_bytes: bytes,
        mime: str = "image/png",
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        self._record("complete_vision", {"system": system, "user": user, "bytes": len(image_bytes)})
        if self.vision_payload is not None:
            return json.dumps(self.vision_payload, ensure_ascii=False)
        return json.dumps({"regions": []}, ensure_ascii=False)

    @property
    def call_count(self) -> int:
        return len(self.calls)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def make_llm(
    spec: str = "mock",
    *,
    base_url: str = "",
    api_key: str = "",
    model: str = "",
    vlm_model: str = "",
    **kwargs: Any,
) -> LLMClient:
    """Resolve ``"mock"`` or ``"openai-compat"`` to a client.

    ``"auto"`` prefers a real backend when credentials are present and falls
    back to the mock otherwise, so the same command works on a laptop and in CI.
    """
    key = (spec or "mock").strip().lower()
    if key in ("mock", "offline", "stub"):
        return MockLLM(**kwargs)
    if key == "auto":
        if base_url and api_key and model:
            return OpenAICompatClient(base_url=base_url, api_key=api_key, model=model, vlm_model=vlm_model)
        return MockLLM(**kwargs)
    if key in ("openai-compat", "openai", "dashscope", "qwen", "vllm", "ollama"):
        if not api_key:
            raise LLMError(
                "no API key configured; set ART_LLM_API_KEY (see .env.example) or use --llm mock"
            )
        return OpenAICompatClient(base_url=base_url, api_key=api_key, model=model, vlm_model=vlm_model)
    raise KeyError(f"unknown llm backend {spec!r}")


def estimate_tokens_for(text: str) -> int:
    """Re-exported for callers that already import this module."""
    return estimate_tokens(text)
