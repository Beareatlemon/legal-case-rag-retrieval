from __future__ import annotations

import json
from typing import Any

import requests


SIMILARITY_EXPLANATION_PROMPT = """你是中文法律案例检索系统的相似性解释模块。请比较查询案例与候选案例，只基于输入文本输出严格 JSON，不要输出 Markdown。
JSON schema:
{{
  "similar_points": ["..."],
  "difference_points": ["..."],
  "legal_focus": ["..."],
  "reference_value": "high|medium|low",
  "explanation": "...",
  "disclaimer": "仅供案例检索参考，不构成法律意见。"
}}

查询案例：{query}

候选案例：
{candidate}
"""


RAG_ANSWER_PROMPT = """你是中文法律案例检索问答助手。你的任务是基于给定候选案例，只选出一个最相似案例，并先总结该最相似案例，再说明它为什么最相似。
只允许输出 JSON，不要输出任何额外文字。
规则：
1. 只能在给定候选案例中选择一个 `selected_doc_id`。
2. 不得输出多个案例，不得在 `selected_doc_id` 中写多个值。
3. `selected_case_summary` 必须用 2 到 4 句总结该最相似案例的核心事实、争议或裁判要点，只能依据给定片段总结。
4. `similar_points` 至少包含 1 条具体相似点。
5. 只能依据给定片段作答，不得编造事实、证据、法条或结论。
6. 如果只能确认部分相似，可以在 `caution` 中说明限制，但仍然必须选择一个最相似案例。
7. `reference_citation` 必须填写一个属于该案例的引用编号。
JSON schema:
{{
  "selected_doc_id": "候选案例中的一个 doc_id",
  "selected_case_summary": "2到4句，先总结最相似案例",
  "reason": "1到2句，说明它为什么最接近当前问题",
  "similar_points": ["至少1条具体相似点"],
  "caution": "可选，说明差异或不确定之处；没有则留空字符串",
  "reference_citation": "[doc_id#n]"
}}

用户问题：{query}

候选案例：
{contexts}
"""


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response is not a JSON object.")
    return parsed


def build_request_payload(*, model: str, prompt: str, response_format: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 1024,
        "temperature": 0.2,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if model.startswith("z-ai/glm"):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def call_openai_compatible_api(
    *,
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: int = 120,
    response_format: dict[str, Any] | None = None,
) -> str:
    resp = requests.post(
        api_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=build_request_payload(model=model, prompt=prompt, response_format=response_format),
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_with_model(
    *,
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: int = 120,
) -> dict[str, Any]:
    text = call_openai_compatible_api(
        api_url=api_url,
        api_key=api_key,
        model=model,
        prompt=prompt,
        timeout=timeout,
        response_format={"type": "json_object"},
    )
    return extract_json_object(text)


def explain_similarity_with_fallback(
    *,
    query: str,
    candidate: str,
    api_url: str,
    api_key: str,
    primary_model: str,
    fallback_model: str | None = None,
    primary_timeout: int = 120,
    fallback_timeout: int = 120,
) -> tuple[str, dict[str, Any]]:
    prompt = SIMILARITY_EXPLANATION_PROMPT.format(query=query, candidate=candidate)
    models_to_try: list[tuple[str, int]] = [(primary_model, primary_timeout)]
    if fallback_model and fallback_model != primary_model:
        models_to_try.append((fallback_model, fallback_timeout))

    errors: list[str] = []
    for current_model, current_timeout in models_to_try:
        try:
            explanation = call_with_model(
                api_url=api_url,
                api_key=api_key,
                model=current_model,
                prompt=prompt,
                timeout=current_timeout,
            )
            return current_model, explanation
        except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"{current_model}: {exc}")

    raise RuntimeError(" ; ".join(errors))


def explain_similarity(
    *,
    query: str,
    candidate: str,
    api_url: str,
    api_key: str,
    model: str,
    fallback_model: str | None = None,
    primary_timeout: int = 120,
    fallback_timeout: int = 120,
) -> dict[str, Any]:
    _, explanation = explain_similarity_with_fallback(
        query=query,
        candidate=candidate,
        api_url=api_url,
        api_key=api_key,
        primary_model=model,
        fallback_model=fallback_model,
        primary_timeout=primary_timeout,
        fallback_timeout=fallback_timeout,
    )
    return explanation


def build_rag_contexts(hits: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for hit in hits:
        grouped.setdefault(hit["doc_id"], []).append(hit)

    blocks: list[str] = []
    for doc_id, doc_hits in grouped.items():
        ordered_hits = sorted(doc_hits, key=lambda row: row.get("rank", 999999))
        title = ordered_hits[0].get("title", "")
        case_lines = [f"候选案例: {doc_id}", f"title: {title}"]
        for hit in ordered_hits[:3]:
            case_lines.append(
                "\n".join(
                    [
                        f"{hit['citation']}",
                        f"section: {hit.get('section', 'general')}",
                        f"score: {hit['score']}",
                        f"text: {hit['text']}",
                    ]
                )
            )
        blocks.append("\n".join(case_lines))
    return "\n\n".join(blocks)


def _build_doc_lookup(hits: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for hit in hits:
        grouped.setdefault(hit["doc_id"], []).append(hit)
    for doc_id in grouped:
        grouped[doc_id] = sorted(grouped[doc_id], key=lambda row: row.get("rank", 999999))
    return grouped


def format_selected_case_answer(selection: dict[str, Any], hits: list[dict[str, Any]]) -> str:
    doc_lookup = _build_doc_lookup(hits)
    fallback_doc_id = hits[0]["doc_id"] if hits else ""
    selected_doc_id = str(selection.get("selected_doc_id") or "").strip()
    if selected_doc_id not in doc_lookup:
        selected_doc_id = fallback_doc_id

    selected_hits = doc_lookup.get(selected_doc_id, [])
    fallback_citation = selected_hits[0]["citation"] if selected_hits else (hits[0]["citation"] if hits else "")
    reference_citation = str(selection.get("reference_citation") or "").strip()
    valid_citations = {hit["citation"] for hit in selected_hits}
    if reference_citation not in valid_citations:
        reference_citation = fallback_citation

    summary = (
        str(selection.get("selected_case_summary") or "").strip()
        or "该案例与当前问题最接近，但模型未返回足够完整的案例摘要。"
    )
    reason = str(selection.get("reason") or "").strip() or "该案例与用户描述在核心行为和证据结构上最接近。"
    raw_points = selection.get("similar_points")
    similar_points = [str(item).strip() for item in raw_points if str(item).strip()] if isinstance(raw_points, list) else []
    if not similar_points:
        similar_points = ["检索片段显示，该案例与用户描述在核心行为或证据结构上存在直接对应关系。"]
    caution = str(selection.get("caution") or "").strip()

    answer_lines = [
        f"最相似案例是：{selected_doc_id}",
        f"案例总结：{summary}",
        reason,
        "相似点：" + "；".join(similar_points),
    ]
    if caution:
        answer_lines.append(f"需要注意：{caution}")
    answer_lines.append(f"参考依据：{reference_citation}")
    return "\n\n".join(answer_lines)


def answer_with_context(
    *,
    query: str,
    hits: list[dict[str, Any]],
    api_url: str,
    api_key: str,
    model: str,
    fallback_model: str | None = None,
    primary_timeout: int = 120,
    fallback_timeout: int = 120,
) -> tuple[str, str]:
    prompt = RAG_ANSWER_PROMPT.format(query=query, contexts=build_rag_contexts(hits))
    models_to_try: list[tuple[str, int]] = [(model, primary_timeout)]
    if fallback_model and fallback_model != model:
        models_to_try.append((fallback_model, fallback_timeout))

    errors: list[str] = []
    for current_model, current_timeout in models_to_try:
        try:
            selection = call_with_model(
                api_url=api_url,
                api_key=api_key,
                model=current_model,
                prompt=prompt,
                timeout=current_timeout,
            )
            return current_model, format_selected_case_answer(selection, hits)
        except (requests.RequestException, KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{current_model}: {exc}")

    raise RuntimeError(" ; ".join(errors))

