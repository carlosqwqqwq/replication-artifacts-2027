"""小模型根级 CaseRewriteHint 解析与 provenance 记录。"""

import json
import math
import os
import re
import time
from contextlib import contextmanager
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from framework._util import canonical_digest, is_sha256_digest

_REWRITE_HINT_FIELDS = frozenset({"edits", "provenance"})
_REWRITE_EDIT_FIELDS = frozenset({
    "source_span", "source_path", "operator", "payload", "candidate_id", "source_text", "source_sha256",
})
_REWRITE_PAYLOAD_FIELDS = frozenset({"mnemonic", "operands", "raw_word_hex"})
_SELECTION_METADATA_FIELDS = frozenset({
    "confidence", "explanation", "rationale", "reason",
})
_MODEL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"candidate_id": {"type": "string"}},
    "required": ["candidate_id"],
    "additionalProperties": False,
}
_CANDIDATE_ID_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", re.IGNORECASE)


@contextmanager
def _model_cache_lock(path: Path, deadline: float | None):
    """Serialize identical local model requests without making the campaign a gate.

    Matrix target workers are separate processes.  A small file lock lets the
    first worker fill one response while the others reuse it; if the lock is
    busy past the caller's deadline the caller proceeds independently.
    """
    handle = None
    acquired = False
    try:
        import fcntl

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.with_suffix(path.suffix + ".lock").open("a+")
        except OSError:
            yield False
            return
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
        yield acquired
    finally:
        if handle is not None:
            if acquired:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            handle.close()


def _model_cache_read(path: Path) -> str | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    content = value.get("content") if isinstance(value, Mapping) else None
    return content if isinstance(content, str) and content.strip() else None


def _model_cache_write(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        # The cache is an optimization.  A read-only /tmp must never change
        # the semantic result of the model route.
        return


def _reject_json_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse(value):
    value = re.sub(
        r'(?P<string>"(?:\\.|[^"\\])*")|(?P<hex>(?<![A-Za-z0-9_])[+-]?0[xX][0-9a-fA-F]+\b)',
        lambda match: match.group("string") or str(int(match.group("hex"), 0)),
        value.strip().lstrip("\ufeff").strip(),
    )
    decoder = json.JSONDecoder(
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    if value.startswith(("[", "{")):
        try:
            parsed, end = decoder.raw_decode(value)
        except ValueError:
            return None
        return parsed if not value[end:].strip() else None
    for start, char in enumerate(value):
        if char not in "[{":
            continue
        try:
            return decoder.raw_decode(value, start)[0]
        except ValueError:
            continue
    try:
        parsed = decoder.decode(value)
    except ValueError:
        return None
    return parsed if type(parsed) is int else None


def parse_s1(text):
    """提取普通或围栏 JSON，兼容十六进制值。"""
    if not isinstance(text, str) or not text:
        return None
    candidates = re.findall(
        r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL
    ) or [text]
    for candidate in reversed(candidates):
        if (parsed := _parse(candidate)) is not None:
            return parsed
    if candidates != [text] and (parsed := _parse(text)) is not None:
        return parsed
    return [] if not any(candidate.strip().lstrip("\ufeff").strip() for candidate in candidates) else None


def _rewrite_provenance(provenance):
    if not isinstance(provenance, Mapping):
        raise ValueError("rewrite provenance must be an object")
    result = {
        name: provenance.get(name)
        for name in ("raw_sha256", "candidate_pool_digest")
    }
    if any(not is_sha256_digest(value) for value in result.values()):
        raise ValueError("rewrite provenance requires raw_sha256 and candidate_pool_digest")
    return result


def rewrite_hint_edits(hint):
    if not isinstance(hint, Mapping):
        return ()
    edits = hint.get("edits")
    return edits if isinstance(edits, list) else ()


def rewrite_candidate_id(edit):
    if not isinstance(edit, Mapping) or any(
        key not in edit for key in ("source_span", "operator", "payload")
    ):
        raise ValueError("rewrite candidate fields are required")
    value = {
        key: edit[key] for key in ("source_span", "operator", "payload")
    }
    if "source_path" in edit:
        value["source_path"] = edit["source_path"]
    return canonical_digest(value)


def _canonical_edit(edit):
    return {
        "source_span": dict(edit["source_span"]),
        "operator": edit["operator"],
        "payload": dict(edit["payload"]),
        **{
            name: edit[name] for name in ("source_path", "source_text", "source_sha256")
            if name in edit
        },
        "candidate_id": edit.get("candidate_id") or rewrite_candidate_id(edit),
    }


def _rewrite_edit_error(edit):
    if not isinstance(edit, Mapping):
        return "schema-error:invalid-edit"
    unknown = sorted(map(str, set(edit) - _REWRITE_EDIT_FIELDS))
    if unknown:
        return "schema-error:unknown-fields=" + ",".join(unknown)
    missing = sorted(
        field for field in ("source_span", "operator", "payload") if field not in edit
    )
    if missing:
        return "schema-error:missing-fields=" + ",".join(missing)
    if "source_path" in edit:
        source_path = edit["source_path"]
        parts = source_path.replace("\\", "/").split("/") if isinstance(source_path, str) else ()
        if (not isinstance(source_path, str) or not source_path.strip()
                or source_path.startswith(("/", "\\"))
                or re.match(r"^[A-Za-z]:", source_path)
                or any(part in {"", ".", ".."} for part in parts)):
            return "schema-error:invalid-source-path"
    span = edit["source_span"]
    if (not isinstance(span, Mapping)
            or set(span) != {"start", "end", "start_col", "end_col"}
            or any(isinstance(span[name], bool) or not isinstance(span[name], int)
                   for name in span)
            or span["end"] < span["start"] or span["start"] < 1
            or span["start_col"] < 0 or span["end_col"] < 0
            or edit.get("operator") == "replace"
            and span["end_col"] <= span["start_col"]):
        return "schema-error:invalid-source-span"
    operator = edit["operator"]
    if not isinstance(operator, str):
        return f"action-reject:operator-not-allowed={operator}"
    payload = edit["payload"]
    if operator == "replace":
        if (not isinstance(payload, Mapping)
                or not {"mnemonic", "operands"} <= set(payload)
                or not set(payload) <= _REWRITE_PAYLOAD_FIELDS
                or not isinstance(payload["mnemonic"], str) or not payload["mnemonic"].strip()
                or not isinstance(payload["operands"], list)
                or not 0 <= len(payload["operands"]) <= 5
                or any(not isinstance(item, str) or not item.strip() for item in payload["operands"])
                or "raw_word_hex" in payload
                and (not isinstance(payload["raw_word_hex"], str)
                     or re.fullmatch(r"0x[0-9a-fA-F]+", payload["raw_word_hex"]) is None)):
            return "schema-error:invalid-scalar-payload"
    elif operator == "replace_fragment":
        if (not isinstance(payload, Mapping) or set(payload) != {"lines"}
                or not isinstance(payload["lines"], list) or len(payload["lines"]) < 2
                or len(payload["lines"]) != span["end"] - span["start"] + 1
                or any(not isinstance(item, str) for item in payload["lines"])
                or not any(item.strip() for item in payload["lines"])):
            return "schema-error:invalid-fragment-payload"
    else:
        return f"action-reject:operator-not-allowed={operator}"
    candidate_id = edit.get("candidate_id")
    if candidate_id is not None and (
        not is_sha256_digest(candidate_id) or candidate_id != rewrite_candidate_id(edit)
    ):
        return "schema-error:invalid-candidate-id"
    if "source_text" in edit and not isinstance(edit["source_text"], str):
        return "schema-error:invalid-source-text"
    if "source_sha256" in edit and not is_sha256_digest(edit["source_sha256"]):
        return "schema-error:invalid-source-sha256"
    return None


def _rewrite_hint_error(hint, provenance):
    if not isinstance(hint, Mapping):
        return "schema-error:rewrite hint must be an object"
    fields = set(hint)
    unknown = sorted(map(str, fields - _REWRITE_HINT_FIELDS))
    if unknown:
        return "schema-error:unknown-fields=" + ",".join(unknown)
    if "edits" not in fields:
        return "schema-error:missing-fields=edits"
    edits = rewrite_hint_edits(hint)
    if not edits:
        return "schema-error:invalid-edits"
    if reason := next((_rewrite_edit_error(edit) for edit in edits), None):
        return reason
    ordered = sorted(
        edits,
        key=lambda edit: (
            edit.get("source_path", ""),
            edit["source_span"]["start"], edit["source_span"]["start_col"],
            edit["source_span"]["end"], edit["source_span"]["end_col"],
        ),
    )
    if any(
        (right["source_span"]["start"], right["source_span"]["start_col"])
        < (left["source_span"]["end"], left["source_span"]["end_col"])
        for left, right in zip(ordered, ordered[1:])
        if left.get("source_path") == right.get("source_path")
    ):
        return "schema-error:overlapping-edits"
    try:
        json.dumps(hint, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return "schema-error:rewrite hint must be JSON-serializable"
    if "provenance" not in hint:
        return None
    try:
        supplied = _rewrite_provenance(hint["provenance"])
    except ValueError:
        return "schema-error:invalid-provenance"
    return "schema-error:provenance-mismatch" if any(
        supplied[name] != provenance[name]
        for name in ("raw_sha256", "candidate_pool_digest")
    ) else None


def _ollama_json(endpoint, path, payload, timeout):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        endpoint + path, data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(f"ollama-http-{error.code}") from None
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"ollama-connection-{type(error).__name__}") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("ollama-invalid-json-response") from None


def _openai_compatible_json(endpoint, payload, timeout):
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("RQ1_SMALL_MODEL_OPENAI_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        endpoint + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(f"openai-compatible-http-{error.code}") from None
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(
            f"openai-compatible-connection-{type(error).__name__}"
        ) from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("openai-compatible-invalid-json-response") from None


def _model_candidate_limit(value=None):
    raw = os.environ.get("RQ1_SMALL_MODEL_MAX_CANDIDATES", "32") if value is None else value
    try:
        limit = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("RQ1_SMALL_MODEL_MAX_CANDIDATES must be a non-negative integer") from error
    if isinstance(raw, bool) or limit < 0:
        raise ValueError("RQ1_SMALL_MODEL_MAX_CANDIDATES must be a non-negative integer")
    return limit


def _candidate_bucket(candidate):
    operator = candidate.get("operator") if isinstance(candidate, Mapping) else None
    span = candidate.get("source_span") if isinstance(candidate, Mapping) else None
    if not isinstance(span, Mapping) or type(span.get("start")) is not int:
        return None
    if operator == "replace":
        return ("replace", 1)
    if operator == "replace_fragment" and type(span.get("end")) is int:
        length = max(2, span["end"] - span["start"] + 1)
        return ("replace_fragment", min(length, 3))
    return None


def _uniform_candidate_view(candidates, limit):
    if limit == 0 or limit >= len(candidates):
        return list(candidates)
    if limit == 1:
        indexes = (0,)
    else:
        indexes = tuple(dict.fromkeys(
            (offset * (len(candidates) - 1) + (limit - 1) // 2) // (limit - 1)
            for offset in range(limit)
        ))
    return [candidates[index] for index in indexes]


def _model_candidate_view(candidates, limit):
    """用固定语义预算覆盖单指令、两指令和三指令候选。"""
    if limit == 0 or limit >= len(candidates):
        return list(candidates)
    buckets = {}
    for candidate in candidates:
        bucket = _candidate_bucket(candidate)
        if bucket is None:
            return _uniform_candidate_view(candidates, limit)
        buckets.setdefault(bucket, []).append(candidate)
    if len(buckets) <= 1:
        return _uniform_candidate_view(candidates, limit)
    order = sorted(buckets, key=lambda item: (item[1], item[0]))
    allocation = {bucket: 0 for bucket in order}
    remaining = limit
    while remaining:
        added = False
        for bucket in order:
            if allocation[bucket] >= len(buckets[bucket]):
                continue
            allocation[bucket] += 1
            remaining -= 1
            added = True
            if not remaining:
                break
        if not added:
            break
    views = [
        _uniform_candidate_view(buckets[bucket], allocation[bucket])
        for bucket in order
    ]
    selected = []
    for index in range(max(allocation.values(), default=0)):
        for view in views:
            if index < len(view) and len(selected) < limit:
                selected.append(view[index])
    return selected


def _candidate_lines(value):
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, list):
        return [line.strip() for line in value if isinstance(line, str) and line.strip()]
    return []


def _model_candidate_record(candidate):
    """把完整候选压成模型可读的序列摘要，保留 ID 作为唯一绑定键。"""
    if not isinstance(candidate, Mapping) or _candidate_bucket(candidate) is None:
        return candidate
    span = candidate["source_span"]
    payload = candidate.get("payload")
    operator = candidate["operator"]
    before = _candidate_lines(candidate.get("source_text"))
    if operator == "replace" and isinstance(payload, Mapping):
        mnemonic = payload.get("mnemonic")
        operands = payload.get("operands")
        after = [f"{mnemonic} {', '.join(operands)}"] if (
            isinstance(mnemonic, str) and isinstance(operands, list)
            and all(isinstance(item, str) for item in operands)
        ) else []
    else:
        after = _candidate_lines(payload.get("lines") if isinstance(payload, Mapping) else None)
    record = {
        "candidate_id": candidate["candidate_id"],
        "operator": operator,
        "source_span": dict(span),
        "span_length": span["end"] - span["start"] + 1
        if type(span.get("end")) is int else 1,
        "before": before,
        "after": after,
    }
    if candidate.get("source_path"):
        record["source_path"] = candidate["source_path"]
    return record


def _model_option_int(name, default, minimum=1):
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _candidate_id_selection(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        return None
    for name in ("candidate_id", "selected_candidate_id"):
        if name in value and set(value) <= {name} | _SELECTION_METADATA_FIELDS:
            return value[name]
    return None


def _resolve_candidate(candidate_id, candidates):
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        return None, "schema-error:invalid-candidate-id"
    if candidates is None:
        return None, "schema-error:candidate-pool-required"
    matches = tuple(
        candidate for candidate in candidates
        if isinstance(candidate, Mapping) and candidate.get("candidate_id") == candidate_id
    )
    if not matches:
        return None, "schema-error:candidate-id-not-in-pool"
    if len(matches) != 1:
        # A stale duplicate must not shadow a pool entry whose rewrite digest
        # exactly matches the model's content-addressed selection.
        exact_matches = []
        for candidate in matches:
            try:
                if rewrite_candidate_id(candidate) == candidate_id:
                    exact_matches.append(candidate)
            except (KeyError, TypeError, ValueError):
                continue
        if exact_matches:
            return dict(exact_matches[0]), None
        return None, "schema-error:ambiguous-candidate-id"
    return dict(matches[0]), None


def _response_candidate_id(text, candidates):
    """从严格响应或明确的最终句中提取候选 ID；不猜测多候选分析文本。"""
    if not isinstance(text, str) or not isinstance(candidates, Sequence):
        return None
    candidate_ids = {
        candidate.get("candidate_id")
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and isinstance(candidate.get("candidate_id"), str)
    }
    parsed = parse_s1(text)
    structured = []
    if isinstance(parsed, Mapping):
        structured.append(parsed)
        edits = parsed.get("edits")
        if isinstance(edits, Mapping):
            structured.append(edits)
        elif isinstance(edits, list) and len(edits) == 1 and isinstance(edits[0], Mapping):
            structured.append(edits[0])
    for value in structured:
        candidate_id = _candidate_id_selection(value)
        if candidate_id in candidate_ids:
            return candidate_id
        if (
            isinstance(value, Mapping)
            and isinstance(value.get("candidate_id"), str)
            and value.get("candidate_id") in candidate_ids
        ):
            return value["candidate_id"]

    marked = []
    for match in re.finditer(
        r"(?is)(?:selected|select|chosen|choose|pick|answer|final|recommend)"
        r"[^\r\n]{0,160}?\b(?:candidate(?:_id)?\s*)?[`'\"]?"
        r"([0-9a-f]{64})\b",
        text,
    ):
        candidate_id = match.group(1)
        if candidate_id in candidate_ids:
            marked.append(candidate_id)
    marked = tuple(dict.fromkeys(marked))
    if len(marked) == 1:
        return marked[0]

    mentioned = tuple(dict.fromkeys(
        match.group(0) for match in _CANDIDATE_ID_RE.finditer(text)
        if match.group(0) in candidate_ids
    ))
    return mentioned[0] if len(mentioned) == 1 else None


def _bind_candidate_selection(hint, candidates):
    """把模型的短 ID 响应绑定回受信任的完整候选。"""
    if not isinstance(hint, Mapping):
        return hint, None
    if {"source_span", "operator", "payload"} <= set(hint):
        return {"edits": [dict(hint)]}, None
    for name in ("candidate_id", "selected_candidate_id"):
        if name in hint and set(hint) <= {name} | _SELECTION_METADATA_FIELDS:
            candidate, error = _resolve_candidate(hint[name], candidates)
            return ({"edits": [candidate]} if candidate is not None else None), error
    edits = hint.get("edits")
    selection = edits if isinstance(edits, Mapping) else (
        edits[0] if isinstance(edits, list) and len(edits) == 1 else None
    )
    candidate_id = _candidate_id_selection(selection)
    if candidate_id is None:
        return hint, None
    candidate, error = _resolve_candidate(candidate_id, candidates)
    if candidate is None:
        return None, error
    return {**hint, "edits": [candidate]}, None


def request_case_rewrite(
    candidates, *, model=None, max_candidates=None, timeout_seconds=None,
    cache_key=None, cache_dir=None,
):
    """从配置的小模型 API 请求一个受候选池约束的 CaseRewriteHint。"""
    request_mode = os.environ.get(
        "RQ1_SMALL_MODEL_REQUEST_MODE", "ollama",
    ).strip().lower()
    if request_mode not in {"ollama", "openai-compatible"}:
        raise ValueError(
            "RQ1_SMALL_MODEL_REQUEST_MODE must be ollama or openai-compatible"
        )
    model = str(model or os.environ.get("RQ1_SMALL_MODEL", "")).strip()
    if not model:
        return None
    for prefix in ("ollama/", "openai-compatible/"):
        if model.startswith(prefix):
            model = model.removeprefix(prefix)
            break
    if request_mode == "openai-compatible":
        model = os.environ.get("RQ1_SMALL_MODEL_OPENAI_MODEL", "").strip()
        if not model:
            raise ValueError("RQ1_SMALL_MODEL_OPENAI_MODEL is empty")
    if not model or not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("model name or candidate pool is invalid")
    if any(
        not isinstance(candidate, Mapping)
        or not isinstance(candidate.get("candidate_id"), str)
        or not candidate["candidate_id"].strip()
        for candidate in candidates
    ):
        raise ValueError("model candidate pool is invalid")
    fields = (
        "candidate_id", "source_span", "source_path", "operator", "payload",
        "source_text", "source_sha256",
    )
    full_candidates = [
        {name: candidate[name] for name in fields if name in candidate}
        for candidate in candidates if isinstance(candidate, Mapping)
    ]
    compact_candidates = _model_candidate_view(
        full_candidates, _model_candidate_limit(max_candidates),
    )
    if not compact_candidates:
        raise ValueError("model candidate pool is empty")
    model_records = [_model_candidate_record(candidate) for candidate in compact_candidates]
    bucket_counts = {}
    for candidate in full_candidates:
        bucket = _candidate_bucket(candidate)
        if bucket is not None:
            name = f"{bucket[0]}:{bucket[1]}"
            bucket_counts[name] = bucket_counts.get(name, 0) + 1
    if request_mode == "ollama":
        endpoint = os.environ.get(
            "OLLAMA_HOST", "http://133.133.135.123:11434",
        ).rstrip("/")
    else:
        endpoint = os.environ.get("RQ1_SMALL_MODEL_OPENAI_BASE_URL", "").strip().rstrip("/")
        if endpoint.endswith("/chat/completions"):
            endpoint = endpoint.removesuffix("/chat/completions")
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("RQ1_SMALL_MODEL_OPENAI_BASE_URL must be an HTTP URL")
    timeout = max(1, int(os.environ.get("RQ1_SMALL_MODEL_TIMEOUT", "120")))
    deadline = None
    if timeout_seconds is None:
        # Keep direct library callers bounded too.  The per-request timeout
        # remains configurable; this is only the total budget across tags,
        # the first response, and retries.
        try:
            timeout_seconds = float(os.environ.get("RQ1_SMALL_MODEL_BUDGET", "120"))
        except (TypeError, ValueError):
            timeout_seconds = 45.0
    if timeout_seconds is not None:
        try:
            budget = float(timeout_seconds)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("timeout_seconds must be non-negative") from error
        if not math.isfinite(budget) or budget < 0:
            raise ValueError("timeout_seconds must be non-negative")
        deadline = time.monotonic() + budget

    def request_timeout(cap: float) -> float:
        if deadline is None:
            return cap
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"{request_mode}-budget-exhausted")
        return max(0.05, min(cap, remaining))

    expected_digest = os.environ.get("RQ1_SMALL_MODEL_DIGEST", "").strip()
    model_digest = None
    if request_mode == "ollama":
        tags = _ollama_json(endpoint, "/api/tags", None, request_timeout(min(timeout, 10)))
        if not isinstance(tags, Mapping):
            raise RuntimeError("ollama-invalid-tags-response")
        tag_models = tags.get("models")
        if not isinstance(tag_models, list):
            raise RuntimeError("ollama-invalid-tags-response")
        models = {
            item.get("name"): item.get("digest")
            for item in tag_models if isinstance(item, Mapping)
        }
        if model not in models:
            raise RuntimeError("ollama-model-not-listed")
        model_digest = models[model]
        if expected_digest and model_digest != expected_digest:
            raise RuntimeError("ollama-model-digest-mismatch")
    elif expected_digest:
        raise RuntimeError("openai-compatible-model-digest-unavailable")
    cache_root_value = cache_dir or os.environ.get("RQ1_SMALL_MODEL_CACHE_DIR")
    cache_path = None
    if cache_root_value:
        cache_root = Path(str(cache_root_value)).expanduser()
        cache_digest = canonical_digest({
            "schema": "rq1-small-model-selection-cache-v1",
            "cache_key": cache_key,
            "request_mode": request_mode,
            "endpoint": endpoint,
            "model": model,
            "model_digest": model_digest,
            "candidate_pool_digest": canonical_digest(full_candidates),
            "candidate_view_digest": canonical_digest(model_records),
            "max_candidates": _model_candidate_limit(max_candidates),
        })
        cache_path = cache_root / f"{cache_digest}.json"
    prompt = json.dumps({
        "instruction": (
            "Choose exactly one candidate_id from candidates. Return only a JSON "
            "object with exactly one key, candidate_id. Copy its value verbatim. "
            "Every candidate is already a valid executed rewrite. Prefer a "
            "replace_fragment with span_length greater than 1 when its before "
            "and after lines form one contiguous sequence; otherwise choose a "
            "single instruction. Do not invent an ID or return an explanation."
        ),
        "candidate_pool_size": len(full_candidates),
        "candidate_view_size": len(compact_candidates),
        "candidate_view": "semantic-budget-v1",
        "candidate_bucket_counts": bucket_counts,
        "candidates": model_records,
    }, ensure_ascii=False, separators=(",", ":"))
    # The model returns one short ID object; a large generation budget invites
    # small models to spend the response on analysis before emitting JSON.
    num_predict = _model_option_int("RQ1_SMALL_MODEL_NUM_PREDICT", 256)
    num_ctx = _model_option_int("RQ1_SMALL_MODEL_NUM_CTX", 8192)
    try:
        temperature = float(os.environ.get("RQ1_SMALL_MODEL_TEMPERATURE", "0"))
    except (TypeError, ValueError) as error:
        raise ValueError("RQ1_SMALL_MODEL_TEMPERATURE must be finite") from error
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("RQ1_SMALL_MODEL_TEMPERATURE must be finite and non-negative")
    retries = _model_option_int("RQ1_SMALL_MODEL_RETRIES", 2, minimum=0)
    retry_ids = [candidate["candidate_id"] for candidate in compact_candidates]
    retry_prompt = json.dumps({
        "instruction": "Return only one JSON object: {\"candidate_id\":\"ID\"}.",
        "candidate_ids": retry_ids,
    }, ensure_ascii=False, separators=(",", ":"))

    def request_selection(
        system_prompt: str, user_prompt: str, token_limit: int, context_limit: int,
    ):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if request_mode == "ollama":
            response = _ollama_json(endpoint, "/api/chat", {
                "model": model,
                "messages": messages,
                "stream": False,
                "think": False,
                "format": _MODEL_RESPONSE_SCHEMA,
                "options": {
                    "temperature": temperature,
                    "num_predict": token_limit,
                    "num_ctx": context_limit,
                },
            }, request_timeout(timeout))
            message = response.get("message") if isinstance(response, Mapping) else None
        else:
            response = _openai_compatible_json(endpoint, {
                "model": model,
                "messages": messages,
                "stream": False,
                "temperature": temperature,
                "max_tokens": token_limit,
                "truncate_prompt_tokens": max(1, context_limit - token_limit),
                "truncation_side": "left",
                "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False},
            }, request_timeout(timeout))
            choices = response.get("choices") if isinstance(response, Mapping) else None
            message = (
                choices[0].get("message")
                if isinstance(choices, list) and choices and isinstance(choices[0], Mapping)
                else None
            )
        return message.get("content") if isinstance(message, Mapping) else None

    def select_content() -> str | None:
        content = request_selection(
            "You return strict JSON only.", prompt, num_predict, num_ctx,
        )
        if isinstance(content, str) and content.strip() \
                and _response_candidate_id(content, compact_candidates) is not None:
            return content
        for _ in range(retries):
            retry_content = request_selection(
                "Output exactly one JSON object and nothing else.",
                retry_prompt, min(num_predict, 128), min(num_ctx, 4096),
            )
            if isinstance(retry_content, str) and retry_content.strip() \
                    and _response_candidate_id(retry_content, compact_candidates) is not None:
                return retry_content
        return None

    if cache_path is not None:
        with _model_cache_lock(cache_path, deadline) as acquired:
            cached = _model_cache_read(cache_path)
            if cached is not None and _response_candidate_id(cached, compact_candidates) is not None:
                return cached
            content = select_content()
            if content is not None and acquired:
                _model_cache_write(cache_path, content)
            return content
    content = select_content()
    if content is not None:
        return content
    # Never pass an ID outside the visible view to the campaign.  The full
    # pool is intentionally larger than the model view, so returning the raw
    # invalid text here would let the later binder accept an unseen ID.
    return None


def supply_case_rewrites(model_outputs, provenance, *, candidates=None):
    """解析一次性的 CaseRewriteHint；适配层负责候选枚举和改写。"""
    provenance = _rewrite_provenance(provenance)
    if isinstance(model_outputs, (bytes, bytearray)):
        raise ValueError("model_outputs must be text or JSON values")
    hint = parse_s1(model_outputs) if isinstance(model_outputs, str) else model_outputs
    if isinstance(hint, Mapping) and isinstance(hint.get("edits"), Mapping):
        hint = {**hint, "edits": [hint["edits"]]}
    selection_error = None
    if hint is not None:
        hint, selection_error = _bind_candidate_selection(hint, candidates)
    if hint is None and isinstance(model_outputs, str) and candidates is not None:
        candidate_id = _response_candidate_id(model_outputs, candidates)
        if candidate_id is not None:
            hint, selection_error = _bind_candidate_selection(
                {"candidate_id": candidate_id}, candidates,
            )
    if selection_error is not None:
        reason = selection_error
    elif isinstance(model_outputs, str) and hint is None:
        reason = (
            "response-empty"
            if model_outputs.strip().lstrip("\ufeff").strip().lower() in {"", "null"}
            else "schema-error:invalid-response"
        )
    elif hint is None or hint == {} or hint == []:
        reason = "response-empty"
    else:
        reason = _rewrite_hint_error(hint, provenance)
    canonical = None
    if reason is None:
        canonical = {
            "edits": [_canonical_edit(edit) for edit in rewrite_hint_edits(hint)],
            "provenance": dict(provenance),
        }
    accepted = int(canonical is not None)
    return {
        "schema": "case-rewrite-hint-supply-v1",
        "provenance": provenance,
        "accepted": accepted,
        "rejected": 1 - accepted,
        "raw_response": model_outputs,
        "canonical_hint": canonical,
        "status": "accepted" if canonical else "rejected",
        "rejection_reason": reason,
    }
