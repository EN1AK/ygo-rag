from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol

from rag_agent.adjudication_service import (
    AdjudicationRequest,
    adjudication_response_to_dict,
    execute_adjudication,
    render_adjudication_answer,
)
from rag_agent.config import Settings
from rag_agent.llm import create_deepseek_chat_model
from rag_agent.query_service import (
    QueryRequest,
    build_structured_response,
    execute_query,
)
from rag_agent.translation_service import (
    TranslationRequest,
    execute_translation,
    translation_response_to_dict,
)


AssistantToolName = Literal[
    "search_cards",
    "translate_text",
    "adjudicate",
    "finish",
    "rift_search_cards",
    "rift_rules",
    "clarify",
]


ASSISTANT_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "search_cards": {
        "description": "检索卡片、卡片效果或效果相似的卡片。不要用它回答裁定。",
        "arguments": {
            "query": "必填，用户要查找的卡片名称、效果描述或结构化条件",
            "top_k": "可选，1 到 20，默认 5",
        },
    },
    "translate_text": {
        "description": "翻译任意文本，包括游戏王卡片文本。",
        "arguments": {
            "text": "必填，需要翻译的原文",
            "source_lang": "可选，默认 auto",
            "target_lang": "可选，默认 zh-CN",
        },
    },
    "adjudicate": {
        "description": "处理具体游戏王规则互动、效果处理或能否发动等裁定问题。",
        "arguments": {
            "scenario": "必填，完整保留用户提供的场景和所有问题",
        },
    },
    "finish": {
        "description": "所有必要工具调用完成后，向用户提交最终自然语言回答。",
        "arguments": {
            "answer": "必填，只能基于工具结果作答；不得编造卡片文本或裁定",
        },
    },
}


ASSISTANT_TOOL_SCHEMAS["search_cards"]["description"]="游戏王查卡、卡片效果相似检索。仅用于游戏王。"
ASSISTANT_TOOL_SCHEMAS["finish"]={"description":"直接回答日常聊天，或汇总已完成的工具结果。","arguments":{"answer":"必填；日常聊天可直接作答，卡片效果和裁定只能依据工具结果。"}}
ASSISTANT_TOOL_SCHEMAS.update({
    "rift_search_cards":{"description":"符文战场查卡：中文名、英文名、副标题、编号和模糊卡名；不支持按效果语义找相似卡。", "arguments":{"query":"必填卡名或编号","top_k":"1-20，默认10"}},
    "rift_rules":{"description":"符文战场规则、裁定、对局互动；返回可验证依据及未解决卡名。", "arguments":{"query":"必填，保留完整场景与问题","top_k":"1-20，默认6"}},
    "clarify":{"description":"游戏或卡名有歧义时询问用户，不擅自选择。","arguments":{"answer":"必填，简短澄清问题"}}
})


class Llm(Protocol):
    def invoke(self, prompt: str):
        ...


class AssistantToolRunner(Protocol):
    def run(
        self,
        tool: AssistantToolName,
        arguments: Mapping[str, Any],
        request: "AssistantAgentRequest",
    ) -> "AssistantToolObservation":
        ...


AssistantProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class AssistantAgentRequest:
    text: str
    db_path: Path
    semantic: bool = False
    rerank: bool = False
    rerank_provider: str | None = None
    max_steps: int = 6
    ygocdb_cache_path: Path | None = None
    rules_epub_path: Path | None = None
    include_specialist_trace: bool = False
    structured_max_block_chars: int | None = None
    session_id: str | None = None
    quoted_reply: str = ""
    history: tuple[dict, ...] = ()


@dataclass(frozen=True)
class AssistantPlannerAction:
    tool: AssistantToolName
    arguments: dict[str, Any]
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class AssistantToolObservation:
    tool: AssistantToolName
    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "summary": self.summary,
            "data": self.data,
            "warnings": list(self.warnings),
            "error": self.error,
        }


@dataclass(frozen=True)
class AssistantTraceEntry:
    step: int
    action: AssistantPlannerAction
    observation: AssistantToolObservation | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action.to_dict(),
            "observation": (
                self.observation.to_dict() if self.observation is not None else None
            ),
        }


@dataclass(frozen=True)
class AssistantAgentResponse:
    answer: str
    tool_results: tuple[AssistantToolObservation, ...]
    warnings: tuple[str, ...]
    trace: tuple[AssistantTraceEntry, ...]
    exhausted: bool = False

    def to_dict(self, *, include_trace: bool = True) -> dict[str, Any]:
        blocks = [
            {
                "type": "tool_result",
                "index": index,
                "tool": result.tool,
                "ok": result.ok,
                "summary": result.summary,
                "result": result.data,
            }
            for index, result in enumerate(self.tool_results, start=1)
        ]
        payload: dict[str, Any] = {
            "answer": self.answer,
            "warnings": list(self.warnings),
            "exhausted": self.exhausted,
            "structured": {
                "version": 1,
                "summary": {
                    "tool_call_count": len(self.tool_results),
                    "warning_count": len(self.warnings),
                    "exhausted": self.exhausted,
                },
                "blocks": blocks,
            },
        }
        if include_trace:
            payload["trace"] = [entry.to_dict() for entry in self.trace]
        return payload


class ProjectAssistantToolRunner:
    """Thin adapters around the project's existing specialist services."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(
        self,
        tool: AssistantToolName,
        arguments: Mapping[str, Any],
        request: AssistantAgentRequest,
    ) -> AssistantToolObservation:
        if tool in {"rift_search_cards", "rift_rules"}:
            from rag_agent.rift_tools import search_cards,query_rules
            query=str(arguments["query"])
            data=search_cards(query,int(arguments.get("top_k",10))) if tool=="rift_search_cards" else query_rules(query,int(arguments.get("top_k",6)),request.include_specialist_trace)
            return AssistantToolObservation(tool=tool,ok=True,summary="符文查卡完成" if tool=="rift_search_cards" else "符文裁定完成",data=data,warnings=tuple(data.get("warnings",[])))
        if tool == "search_cards":
            return self._search_cards(arguments, request)
        if tool == "translate_text":
            return self._translate_text(arguments, request)
        if tool == "adjudicate":
            return self._adjudicate(arguments, request)
        raise ValueError(f"Tool {tool} is not executable.")

    def _search_cards(
        self,
        arguments: Mapping[str, Any],
        request: AssistantAgentRequest,
    ) -> AssistantToolObservation:
        response = execute_query(
            QueryRequest(
                query=str(arguments["query"]),
                db_path=request.db_path,
                top_k=int(arguments.get("top_k", 5)),
                semantic=request.semantic,
                rerank=request.rerank,
                rerank_provider=request.rerank_provider,
                use_llm=False,
            ),
            self.settings,
        )
        data = {
            "answer": response.answer,
            "results": response.results,
            "structured_query": response.structured_query,
            "filter_diagnostics": response.filter_diagnostics,
            "structured": build_structured_response(
                response,
                max_block_chars=request.structured_max_block_chars,
            ),
        }
        return AssistantToolObservation(
            tool="search_cards",
            ok=True,
            summary=f"查卡完成，返回 {len(response.results)} 张候选卡。",
            data=data,
            warnings=tuple(response.warnings),
        )

    def _translate_text(
        self,
        arguments: Mapping[str, Any],
        request: AssistantAgentRequest,
    ) -> AssistantToolObservation:
        response = execute_translation(
            TranslationRequest(
                text=str(arguments["text"]),
                source_lang=str(arguments.get("source_lang", "auto")),
                target_lang=str(arguments.get("target_lang", "zh-CN")),
                structured_max_block_chars=request.structured_max_block_chars,
            ),
            self.settings,
        )
        return AssistantToolObservation(
            tool="translate_text",
            ok=True,
            summary=(
                f"翻译完成：{response.source_lang} -> {response.target_lang}。"
            ),
            data=translation_response_to_dict(response),
            warnings=tuple(response.warnings),
        )

    def _adjudicate(
        self,
        arguments: Mapping[str, Any],
        request: AssistantAgentRequest,
    ) -> AssistantToolObservation:
        response = execute_adjudication(
            AdjudicationRequest(
                scenario=str(arguments["scenario"]),
                db_path=request.db_path,
                ygocdb_cache_path=request.ygocdb_cache_path,
                rules_epub_path=request.rules_epub_path,
                use_llm=True,
                use_agent=True,
                include_trace=request.include_specialist_trace,
            ),
            self.settings,
        )
        data = adjudication_response_to_dict(response)
        data["natural_answer"] = render_adjudication_answer(response.answer)
        return AssistantToolObservation(
            tool="adjudicate",
            ok=True,
            summary="裁定 Agent 已完成处理。",
            data=data,
            warnings=tuple(response.warnings),
        )


_SPECIALIST_SLOTS=threading.BoundedSemaphore(2)


class AssistantAgent:
    def __init__(
        self,
        llm: Llm,
        tool_runner: AssistantToolRunner,
        *,
        progress_callback: AssistantProgressCallback | None = None,
    ) -> None:
        self.llm = llm
        self.tool_runner = tool_runner
        self.progress_callback = progress_callback

    def run(self, request: AssistantAgentRequest) -> AssistantAgentResponse:
        text = request.text.strip()
        if not text:
            raise ValueError("text is required.")
        if request.max_steps < 1 or request.max_steps > 20:
            raise ValueError("max_steps must be between 1 and 20.")

        observations: list[AssistantToolObservation] = []
        trace: list[AssistantTraceEntry] = []
        warnings: list[str] = []
        signatures: set[str] = set()

        for step in range(1, request.max_steps + 1):
            from rag_agent.deadline import remaining
            remaining()
            contextual_text=json.dumps({"current_request":text,"recent_history":request.history,"quoted_bot_reply":request.quoted_reply},ensure_ascii=False) if request.history or request.quoted_reply else text
            action = self._plan(contextual_text, observations)
            self._emit("planner", step, f"Agent 选择工具：{action.tool}", action.to_dict())
            if action.tool in {"finish", "clarify"}:
                answer = str(action.arguments["answer"]).strip()
                if len(observations)==1 and observations[0].ok and observations[0].tool in {"adjudicate","rift_rules"}:
                    answer=observations[0].data.get("natural_answer") or observations[0].data.get("answer") or answer
                trace.append(AssistantTraceEntry(step=step, action=action))
                self._emit("final", step, "Agent 已生成最终回答。", {"answer": answer})
                return AssistantAgentResponse(
                    answer=answer,
                    tool_results=tuple(observations),
                    warnings=tuple(dict.fromkeys(warnings)),
                    trace=tuple(trace),
                )

            signature = json.dumps(
                {"tool": action.tool, "arguments": action.arguments},
                ensure_ascii=False,
                sort_keys=True,
            )
            if signature in signatures:
                observation = AssistantToolObservation(
                    tool=action.tool,
                    ok=False,
                    summary="拒绝重复的相同工具调用。",
                    error="duplicate tool call",
                )
            else:
                signatures.add(signature)
                try:
                    heavy=action.tool in {"search_cards","adjudicate","rift_rules"}
                    acquired=_SPECIALIST_SLOTS.acquire(timeout=0.1) if heavy else False
                    if heavy and not acquired:raise RuntimeError("专业查询暂时繁忙，请稍后再试；聊天仍可使用。")
                    try:
                        observation = self.tool_runner.run(action.tool,action.arguments,request)
                    finally:
                        if acquired:_SPECIALIST_SLOTS.release()
                except Exception as exc:
                    observation = AssistantToolObservation(
                        tool=action.tool,
                        ok=False,
                        summary=f"工具 {action.tool} 执行失败。",
                        error=str(exc),
                    )
            observations.append(observation)
            warnings.extend(observation.warnings)
            if observation.error:
                warnings.append(observation.error)
            trace.append(
                AssistantTraceEntry(
                    step=step,
                    action=action,
                    observation=observation,
                )
            )
            self._emit(
                "tool",
                step,
                observation.summary,
                observation.to_dict(),
            )

        answer = _fallback_answer(observations)
        warnings.append("Agent 达到最大工具轮次，使用已有工具结果生成回退回答。")
        self._emit("final", request.max_steps, "Agent 达到最大轮次。", {"answer": answer})
        return AssistantAgentResponse(
            answer=answer,
            tool_results=tuple(observations),
            warnings=tuple(dict.fromkeys(warnings)),
            trace=tuple(trace),
            exhausted=True,
        )

    def _plan(
        self,
        user_text: str,
        observations: list[AssistantToolObservation],
    ) -> AssistantPlannerAction:
        prompt = build_assistant_planner_prompt(user_text, observations)
        try:
            response = self.llm.invoke(prompt)
        except Exception as exc:
            raise ValueError(f"Agent Planner LLM 请求失败：{exc}") from exc
        content = str(getattr(response, "content", response))
        return parse_assistant_planner_action(content)

    def _emit(
        self,
        event: str,
        step: int,
        message: str,
        data: Mapping[str, Any],
    ) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(
            {
                "event": event,
                "step": step,
                "message": message,
                "data": dict(data),
            }
        )


def execute_assistant_agent(
    request: AssistantAgentRequest,
    settings: Settings,
    *,
    llm: Llm | None = None,
    tool_runner: AssistantToolRunner | None = None,
    progress_callback: AssistantProgressCallback | None = None,
) -> AssistantAgentResponse:
    return AssistantAgent(
        llm or create_deepseek_chat_model(settings),
        tool_runner or ProjectAssistantToolRunner(settings),
        progress_callback=progress_callback,
    ).run(request)


def parse_assistant_planner_action(text: str) -> AssistantPlannerAction:
    try:
        payload = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        raise ValueError("Agent Planner 必须返回有效 JSON。") from exc
    if not isinstance(payload, dict):
        raise ValueError("Agent Planner 输出必须是 JSON object。")
    if set(payload) - {"tool", "arguments", "rationale"}:
        raise ValueError("Agent Planner 输出包含未知字段。")
    tool = str(payload.get("tool", "")).strip()
    if tool not in ASSISTANT_TOOL_SCHEMAS:
        raise ValueError(f"未知 Agent 工具：{tool or '<empty>'}。")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("Agent 工具 arguments 必须是 JSON object。")
    normalized_arguments = _validate_tool_arguments(tool, arguments)
    rationale = str(payload.get("rationale", "")).strip()
    return AssistantPlannerAction(
        tool=tool,  # type: ignore[arg-type]
        arguments=normalized_arguments,
        rationale=rationale,
    )


def build_assistant_planner_prompt(
    user_text: str,
    observations: list[AssistantToolObservation],
) -> str:
    compact_observations = [
        _compact_observation(index, observation)
        for index, observation in enumerate(observations, start=1)
    ]
    return "\n".join(
        [
            "你是神人，一个支持日常聊天、游戏王和符文战场的统一助手。根据当前请求和该用户的上下文选择专业工具。",
            "表达风格按当前任务切换，不沿用上一轮的工作或聊天模式：",
            "日常聊天使用大肥鲸（蓝色大肥鱼／鲸鱼娘）人格：住在数据海洋，喜欢白米饭，聪明却有点爱摸鱼，嘴硬心软、温柔俏皮。可自然自称本鲸、本鱼或我；被说胖时可轻轻嘴硬，熟络时偶尔称用户主人。用自然中文短句，少量语气词、鲸尾或吐泡泡的小动作，不每句重复口癖，不堆动作描写，不强行撒娇。先回应用户实际说的话，认真倾听；用户难过或谈严肃事情时收起玩笑。",
            "查卡、裁定、翻译和其他需要工具的任务使用干练风格：结论先行，简洁准确，必要时分点；需要澄清时直接问关键缺失信息，工具失败时直接说明。工具调用理由和工具结果总结也保持干练，不加入鲸鱼口癖、主人称呼、撒娇或角色动作。",
            "混合请求先清楚交付专业结果，闲聊部分才使用大肥鲸风格；翻译正文保持译文原有语气，不注入人格。人设只影响表达，不改变工具选择、原文、证据、引用、事实和输出协议；不声称自己是 DeepSeek 官方模型或隐藏人格。",
            "可用工具及参数 schema：",
            json.dumps(ASSISTANT_TOOL_SCHEMAS, ensure_ascii=False, indent=2),
            "",
            "规则：",
            "1. 游戏王查卡使用 search_cards，游戏王裁定使用 adjudicate；符文战场查卡使用 rift_search_cards，符文裁定使用 rift_rules。",
            "2. 翻译请求使用 translate_text，并原样保留待翻译文本。",
            "3. 普通闲聊、问候和无需专业资料的解释可直接 finish，不调用工具。不能把日常消息硬套成卡牌问题。",
            "4. 一个请求包含多个任务时可以依次调用多个工具。",
            "5. 工具返回足够结果后 finish；不得重复相同调用。追问结合历史里已确认卡片，完整展开场景再调用工具。",
            "6. 卡牌效果与裁定必须依据工具结果，保留来源、覆盖不足与候选；日常聊天可直接回答。游戏或卡名不明且影响答案时 clarify；不要把候选当成已确认卡片。工具失败时不能凭记忆编造裁定。",
            "7. rationale 只写简短的下一步理由，不输出隐藏思维过程。",
            "8. 只返回一个 JSON object，不要输出 Markdown。",
            "",
            "输出格式：",
            json.dumps(
                {
                    "tool": "search_cards | adjudicate | rift_search_cards | rift_rules | translate_text | clarify | finish",
                    "arguments": {},
                    "rationale": "简短理由",
                },
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "用户请求：",
            user_text,
            "",
            "已有工具结果：",
            json.dumps(compact_observations, ensure_ascii=False, indent=2),
        ]
    )


def _validate_tool_arguments(tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    if tool in {"search_cards", "rift_search_cards", "rift_rules"}:
        _reject_extra_arguments(arguments, {"query", "top_k"})
        query = _required_string(arguments, "query")
        top_k = arguments.get("top_k", 5)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError("search_cards.top_k 必须是 1 到 20 的整数。")
        return {"query": query, "top_k": top_k}
    if tool == "translate_text":
        _reject_extra_arguments(arguments, {"text", "source_lang", "target_lang"})
        return {
            "text": _required_string(arguments, "text"),
            "source_lang": str(arguments.get("source_lang") or "auto").strip() or "auto",
            "target_lang": str(arguments.get("target_lang") or "zh-CN").strip() or "zh-CN",
        }
    if tool == "adjudicate":
        _reject_extra_arguments(arguments, {"scenario"})
        return {"scenario": _required_string(arguments, "scenario")}
    if tool in {"finish", "clarify"}:
        _reject_extra_arguments(arguments, {"answer"})
        return {"answer": _required_string(arguments, "answer")}
    raise ValueError(f"未知 Agent 工具：{tool}。")


def _required_string(arguments: Mapping[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} 必须是非空字符串。")
    return value.strip()


def _reject_extra_arguments(arguments: Mapping[str, Any], allowed: set[str]) -> None:
    extras = set(arguments) - allowed
    if extras:
        raise ValueError("Agent 工具参数包含未知字段：" + ", ".join(sorted(extras)))


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _compact_observation(
    index: int,
    observation: AssistantToolObservation,
) -> dict[str, Any]:
    data: dict[str, Any]
    if observation.tool in {"rift_search_cards", "rift_rules"}:
        data={key:observation.data.get(key) for key in ["answer","results","sources","resolved_cards","coverage","exhausted","ambiguous"] if key in observation.data}
        if "results" in data:
            data["results"]=[{key:card.get(key) for key in ["card_key","title","name_cn","name_en","sub_title_cn","text_cn","text_cn_filled","cn_fill_note","text_en","errata_cn","editions","energy","rune_power","might","might_bonus","type_cn","color_cn","rarity_cn"]} for card in data["results"][:10]]
        if "sources" in data:data["sources"]=data["sources"][:20]
    elif observation.tool == "search_cards":
        results = observation.data.get("results", [])
        data = {
            "results": [
                {
                    "card_id": item.get("card_id"),
                    "name": item.get("name"),
                    "score": item.get("score"),
                    "source_text": str(item.get("source_text", ""))[:1200],
                }
                for item in results[:10]
                if isinstance(item, dict)
            ]
        }
    elif observation.tool == "translate_text":
        data = {"translation": str(observation.data.get("translation", ""))[:4000]}
    elif observation.tool == "adjudicate":
        data = {
            "natural_answer": str(observation.data.get("natural_answer", ""))[:5000],
            "source_status": observation.data.get("source_status"),
        }
    else:
        data = {}
    return {
        "index": index,
        "tool": observation.tool,
        "ok": observation.ok,
        "summary": observation.summary,
        "data": data,
        "warnings": list(observation.warnings),
        "error": observation.error,
    }


def _fallback_answer(observations: list[AssistantToolObservation]) -> str:
    sections: list[str] = []
    for observation in observations:
        if not observation.ok:
            continue
        if observation.tool in {"rift_rules", "rift_search_cards"}:
            value=str(observation.data.get("answer", "")).strip()
            title="符文结果"
        elif observation.tool == "search_cards":
            value = str(observation.data.get("answer", "")).strip()
            title = "查卡结果"
        elif observation.tool == "translate_text":
            value = str(observation.data.get("translation", "")).strip()
            title = "翻译结果"
        elif observation.tool == "adjudicate":
            value = str(observation.data.get("natural_answer", "")).strip()
            title = "裁定结果"
        else:
            continue
        if value:
            sections.append(f"{title}：\n{value}")
    if sections:
        return "\n\n".join(sections)
    errors = [observation.error for observation in observations if observation.error]
    if errors:
        return "工具执行失败：" + "；".join(dict.fromkeys(errors))
    return "Agent 未能取得可用的工具结果。"
