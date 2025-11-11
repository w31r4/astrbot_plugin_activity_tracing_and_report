from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, cast

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.message.message_event_result import MessageEventResult
from astrbot.core.star.filter.command import GreedyStr
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

if TYPE_CHECKING:
    from astrbot.core.star.register.star_handler import RegisteringCommandable


@dataclass
class SessionMessage:
    sender_id: str
    sender_name: str
    content: str
    timestamp: datetime
    channel_id: str

    def to_dict(self) -> dict:
        return {
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "channel_id": self.channel_id,
        }

    @staticmethod
    def from_dict(data: dict) -> SessionMessage:
        return SessionMessage(
            sender_id=data["sender_id"],
            sender_name=data["sender_name"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            channel_id=data["channel_id"],
        )


@dataclass
class TargetState:
    target_id: str
    session_id: str
    platform_id: str
    records: deque[SessionMessage] = field(
        default_factory=lambda: deque(),  # FIFO, manual trimming
    )
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime | None = None
    backfilled: bool = False

    def to_dict(self) -> dict:
        """将状态序列化为字典，不包含消息记录。"""
        return {
            "target_id": self.target_id,
            "session_id": self.session_id,
            "platform_id": self.platform_id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
            if self.last_updated
            else None,
            "backfilled": self.backfilled,
        }

    @staticmethod
    def from_dict(data: dict) -> TargetState:
        """从字典反序列化为状态对象。"""
        created_at = datetime.fromisoformat(data["created_at"])
        last_updated = (
            datetime.fromisoformat(data["last_updated"])
            if data.get("last_updated")
            else None
        )
        return TargetState(
            target_id=data["target_id"],
            session_id=data["session_id"],
            platform_id=data["platform_id"],
            created_at=created_at,
            last_updated=last_updated,
            backfilled=data.get("backfilled", False),
        )


@dataclass
class ForecastRule:
    label: str
    target_id: str
    keywords: list[str]
    threshold: float = 0.85
    window_hours: int = 6
    cooldown_minutes: int = 60
    last_triggered: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "target_id": self.target_id,
            "keywords": self.keywords,
            "threshold": self.threshold,
            "window_hours": self.window_hours,
            "cooldown_minutes": self.cooldown_minutes,
            "last_triggered": self.last_triggered.isoformat()
            if self.last_triggered
            else None,
        }

    @staticmethod
    def from_dict(data: dict) -> ForecastRule:
        last = data.get("last_triggered")
        last_dt = (
            datetime.fromisoformat(last).astimezone(timezone.utc) if last else None
        )
        return ForecastRule(
            label=data["label"],
            target_id=data["target_id"],
            keywords=data.get("keywords", []),
            threshold=float(data.get("threshold", 0.85)),
            window_hours=int(data.get("window_hours", 6)),
            cooldown_minutes=int(data.get("cooldown_minutes", 60)),
            last_triggered=last_dt,
        )


@register(
    "astrbot_plugin_activity_tracing_and_report",
    "ZenFun",
    "监听指定群友，构建可检索的时间序列记忆并驱动多智能体分析",
    "0.2.0",
)
class ActivityTracingPlugin(Star):
    """目标式行为监听 + RAG 分析插件。"""

    def __init__(self, context: Context):
        super().__init__(context)
        self.session_histories: dict[str, deque[SessionMessage]] = defaultdict(deque)
        self.session_hash_sequences: dict[str, deque[str]] = defaultdict(deque)
        self.session_hash_sets: dict[str, set[str]] = defaultdict(set)
        self.tracked_targets: dict[str, dict[str, TargetState]] = defaultdict(dict)
        self.user_names: dict[str, str] = {}
        data_root = get_astrbot_data_path()
        self.history_db_path = os.path.join(data_root, "activity_history_index.db")
        self.forecast_store_path = os.path.join(
            data_root,
            "activity_forecast_rules.json",
        )
        self.targets_store_path = os.path.join(
            data_root, "activity_tracked_targets.json"
        )
        self.sessions_store_path = os.path.join(
            data_root, "activity_session_histories.json"
        )
        self.forecast_rules: dict[str, dict[str, ForecastRule]] = defaultdict(dict)

        self.session_history_limit = 6000
        self.target_history_limit = 2000
        self.backfill_limit = 2000
        self.context_span = 2
        self.rag_context_limit = 12
        self.default_snapshot_window = 24

        self.positive_keywords = [
            "开心",
            "顺利",
            "搞定",
            "美味",
            "期待",
            "棒",
            "舒服",
            "升職",
            "爽",
            "上岸",
        ]
        self.negative_keywords = [
            "累",
            "困",
            "焦虑",
            "烦",
            "卡住",
            "崩溃",
            "破防",
            "迟到",
            "堵",
            "请假",
            "感冒",
            "发烧",
        ]
        self.activity_keywords = [
            "吃饭",
            "聚餐",
            "打游戏",
            "上号",
            "排位",
            "开黑",
            "出差",
            "加班",
        ]

    async def initialize(self):
        await self._load_forecast_rules()
        await self._load_session_histories()
        await self._load_tracked_targets()
        # 重建索引
        for session_id, history in self.session_histories.items():
            for msg in history:
                self._rebuild_indexes(session_id, msg)
        for session_id, targets in self.tracked_targets.items():
            for target_id, state in targets.items():
                self._reindex_target(state)

    async def terminate(self):
        """清理内存索引并持久化状态。"""
        await self._save_forecast_rules()
        await self._save_tracked_targets()
        await self._save_session_histories()
        self.session_histories.clear()
        self.session_hash_sequences.clear()
        self.session_hash_sets.clear()
        self.tracked_targets.clear()

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def capture_message_stream(self, event: AstrMessageEvent):
        """被动监听所有消息，归档为 RAG 原子片段。"""
        text = (event.message_str or "").strip()
        if not text:
            return

        sender_id = str(event.get_sender_id())
        if not sender_id or sender_id == str(event.get_self_id()):
            return

        session_id = event.unified_msg_origin
        display_name = event.get_sender_name() or sender_id
        msg = SessionMessage(
            sender_id=sender_id,
            sender_name=display_name,
            content=text,
            timestamp=self._now(),
            channel_id=session_id,
        )
        self.user_names[sender_id] = display_name

        appended = self._append_session_message(session_id, msg)
        if not appended:
            return

        session_targets = self.tracked_targets.get(session_id, {})
        if session_targets:
            target_state = session_targets.get(sender_id)
            if target_state:
                self._record_target_message(target_state, msg)

    @filter.command_group("activity")
    def activity(self):
        """活动追踪"""
        return None

    activity = cast("RegisteringCommandable", activity)

    @activity.group("forecast")  # type: ignore [attr-defined]
    def activity_forecast(self):
        """展望未来"""
        return None

    activity_forecast = cast("RegisteringCommandable", activity_forecast)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @activity_forecast.command("add")  # type: ignore [attr-defined]
    async def forecast_add(
        self,
        event: AstrMessageEvent,
        target: str | None = None,
        label: str | None = None,
        threshold: float = 0.85,
        keywords: GreedyStr = GreedyStr(""),
    ):
        """为目标添加一条展望未来告警规则。"""
        if not label:
            yield event.plain_result("请为规则设置 label，例如 dinner_alert。")
            return
        target_id = self._resolve_target_id(event, target or "", True)
        if not target_id:
            yield event.plain_result("请提供目标 ID 或使用 @ 指定。")
            return
        keyword_text = (keywords or "").strip()
        if not keyword_text:
            yield event.plain_result("请提供用来判断事件的关键词，多个可用空格分隔。")
            return
        word_list = [
            w.strip() for w in keyword_text.replace("，", " ").split() if w.strip()
        ]
        if not word_list:
            yield event.plain_result("关键词解析失败，请重新输入。")
            return
        threshold = max(0.1, min(0.99, float(threshold)))
        rules = self.forecast_rules[event.unified_msg_origin]
        if label in rules:
            yield event.plain_result(f"规则 {label} 已存在，请先删除或更换 label。")
            return
        rules[label] = ForecastRule(
            label=label,
            target_id=target_id,
            keywords=word_list,
            threshold=threshold,
        )
        await self._save_forecast_rules()
        yield event.plain_result(
            "已创建展望未来规则：\n"
            f"- label: {label}\n"
            f"- target: {target_id}\n"
            f"- keywords: {', '.join(word_list)}\n"
            f"- threshold: {threshold:.2f}",
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @activity_forecast.command("rm")  # type: ignore [attr-defined]
    async def forecast_remove(self, event: AstrMessageEvent, label: str | None = None):
        rules = self.forecast_rules.get(event.unified_msg_origin, {})
        if not label:
            yield event.plain_result("请提供要删除的 label。")
            return
        removed = rules.pop(label, None)
        if not removed:
            yield event.plain_result(f"未找到 label 为 {label} 的规则。")
            return
        await self._save_forecast_rules()
        yield event.plain_result(f"已删除规则 {label}。")

    @activity_forecast.command("ls")  # type: ignore [attr-defined]
    async def forecast_list(self, event: AstrMessageEvent):
        rules = self.forecast_rules.get(event.unified_msg_origin, {})
        if not rules:
            yield event.plain_result("当前会话没有展望未来规则。")
            return
        lines = ["展望未来规则："]
        for label, rule in rules.items():
            lines.append(
                f"- {label}: target={rule.target_id}, keywords={','.join(rule.keywords)}, "
                f"threshold={rule.threshold:.2f}, window={rule.window_hours}h",
            )
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @activity.command("watch")  # type: ignore [attr-defined]
    async def activity_watch(self, event: AstrMessageEvent, target: str | None = None):
        """管理员指定监听对象并立即回溯最近 2000 条消息。"""
        session_id = event.unified_msg_origin
        platform_id = event.get_platform_id()
        target_id = self._resolve_target_id(
            event,
            target or "",
            require_explicit=True,
        )
        if not target_id:
            yield event.plain_result("请提供目标 ID 或使用 @ 指定。")
            return

        targets = self.tracked_targets[session_id]
        if target_id in targets:
            yield event.plain_result(f"ID {target_id} 已在监听列表中。")
            return

        state = TargetState(
            target_id=target_id,
            session_id=session_id,
            platform_id=platform_id,
        )
        targets[target_id] = state

        imported = await self._backfill_session_history(session_id, platform_id)
        reindexed = self._reindex_target(state)
        state.backfilled = imported > 0

        yield event.plain_result(
            "监听已开启。\n"
            f"- 目标 ID: {target_id}\n"
            f"- 回溯导入: {imported} 条（需要平台启用消息存档）\n"
            f"- 当前掌握: {reindexed} 条个人发言\n"
            "- 新消息会持续写入临时记忆，不落盘。",
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @activity.command("drop")  # type: ignore [attr-defined]
    async def activity_drop(self, event: AstrMessageEvent, target: str | None = None):
        """停止监听某个群友。"""
        session_id = event.unified_msg_origin
        target_id = self._resolve_target_id(event, target or "", False)
        if not target_id:
            yield event.plain_result("请提供待移除的 ID 或使用 @。")
            return
        targets = self.tracked_targets.get(session_id, {})
        removed = targets.pop(target_id, None)
        if not removed:
            yield event.plain_result(f"未找到 ID {target_id} 的监听任务。")
            return
        yield event.plain_result(f"已停止监听 {target_id}，缓存数据将很快被回收。")

    @activity.command("list")  # type: ignore [attr-defined]
    async def activity_list(self, event: AstrMessageEvent):
        """查看当前会话的所有监听对象。"""
        session_id = event.unified_msg_origin
        targets = self.tracked_targets.get(session_id, {})
        if not targets:
            yield event.plain_result("当前会话尚未监听任何成员。")
            return
        now = self._now()
        lines = ["监听对象："]
        for target_id, state in targets.items():
            total = len(state.records)
            freshness = (
                self._fmt_delta(now - state.last_updated)
                if state.last_updated
                else "未收到发言"
            )
            lines.append(
                f"- ID {target_id} | 记录 {total} 条 | 最近 {freshness} | 回溯{'✓' if state.backfilled else '×'}",
            )
        yield event.plain_result("\n".join(lines))

    @activity.command("snapshot")  # type: ignore [attr-defined]
    async def activity_snapshot(
        self,
        event: AstrMessageEvent,
        target: str | None = None,
        window: int = 24,
    ):
        """获取目标在最近 N 小时内的活动摘要。"""
        session_id = event.unified_msg_origin
        target_id = self._resolve_target_id(event, target or "", False)
        if not target_id:
            yield event.plain_result("请提供目标 ID 或 @ 对方。")
            return
        state = self._get_target_state(session_id, target_id)
        if not state:
            yield event.plain_result("尚未监听该成员，请先使用 /activity watch。")
            return
        window_hours = max(1, min(168, window or self.default_snapshot_window))
        cutoff = self._now() - timedelta(hours=window_hours)
        window_records = [msg for msg in state.records if msg.timestamp >= cutoff]
        if not window_records:
            yield event.plain_result(f"过去 {window_hours} 小时没有该成员的发言。")
            return
        snapshot = self._build_snapshot(window_records, window_hours)
        yield event.plain_result(snapshot)

    @activity.command("rewind")  # type: ignore [attr-defined]
    async def activity_rewind(
        self,
        event: AstrMessageEvent,
        target: str | None = None,
        day: str | None = None,
        query: GreedyStr = GreedyStr(""),
    ):
        """查询某天目标是否提及过某句话或事件。"""
        query_text = (query or "").strip()
        if not query_text:
            yield event.plain_result("请提供需要检索的关键词或事件描述。")
            return
        day_text = (day or "").strip()
        if not day_text:
            yield event.plain_result("请提供日期，格式例如 2024-06-18。")
            return
        target_id = self._resolve_target_id(event, target or "", True)
        if not target_id:
            yield event.plain_result("请提供目标 ID 或使用 @ 指定。")
            return
        try:
            start_local = datetime.strptime(day_text, "%Y-%m-%d")
        except ValueError:
            yield event.plain_result("日期格式不正确，请使用 YYYY-MM-DD。")
            return
        local_tz = datetime.now().astimezone().tzinfo
        start = start_local.replace(tzinfo=local_tz).astimezone(timezone.utc)
        end = start + timedelta(days=1)

        matches = await self._query_history_db(
            event.unified_msg_origin,
            target_id,
            start,
            end,
            query_text,
        )
        if not matches:
            state = self._get_target_state(event.unified_msg_origin, target_id)
            if state:
                matches = self._query_memory_for_day(
                    state.records,
                    start,
                    end,
                    query_text,
                )
        if not matches:
            yield event.plain_result("未找到符合条件的历史记录。")
            return

        name = self.user_names.get(target_id, target_id)
        header = f"【回到过去】{name} 在 {day_text} 提及 '{query_text}' 的记录（共 {len(matches)} 条）"
        lines = [header]
        for rec in matches[:20]:
            local_time = rec["timestamp"].astimezone().strftime("%H:%M")
            snippet = rec["content"]
            if len(snippet) > 80:
                snippet = snippet[:80] + "..."
            lines.append(f"- {local_time} {snippet}")
        if len(matches) > 20:
            lines.append("...（已截断，更多内容可在索引数据库中查询）")
        yield event.plain_result("\n".join(lines))

    @activity.command("ask")  # type: ignore
    async def activity_ask(
        self,
        event: AstrMessageEvent,
        target: str | None = None,
        question: GreedyStr = GreedyStr(""),
    ):
        """向 LLM 发起多智能体推理，回答“某事件发生的可能性”。"""
        raw_question = (question or "").strip()
        if not raw_question:
            yield event.plain_result("请在指令后附上需要分析的问题。")
            return

        session_id = event.unified_msg_origin
        target_id = self._resolve_target_id(event, target or "", False)
        if not target_id:
            yield event.plain_result("请提供目标 ID 或 @ 对方。")
            return
        state = self._get_target_state(session_id, target_id)
        if not state:
            yield event.plain_result("该成员尚未被监听，无法构建 RAG。")
            return

        time_focus = self._infer_time_focus(raw_question)
        rag_chunks = self._build_rag_chunks(
            session_id, target_id, raw_question, time_focus
        )
        if not rag_chunks:
            yield event.plain_result("暂时没有足够的上下文来回答这个问题。")
            return

        coverage = self._describe_coverage(state)
        time_hint = self._format_time_focus(time_focus)
        rag_context = "\n".join(
            f"[{chunk['timestamp']}] {chunk['snippet']}" for chunk in rag_chunks
        )
        system_prompt = (
            "你是多智能体协同分析器，协调“观察员”“时间线校准员”“推理官”三个视角。"
            "阅读提供的原始片段后，分三步输出："
            "1) 观察员梳理事实（列出关键信息及时间）;"
            "2) 时间线校准员评估提问中的时间约束与现有记录的匹配度;"
            "3) 推理官给出该事件发生的可能性（高/中/低 + 理由 + 需要补充的信息）。"
            "禁止捏造上下文外的信息。"
        )
        prompt = (
            f"# 目标\nID {target_id}\n"
            f"# 覆盖\n{coverage}\n"
            f"# 时间提示\n{time_hint}\n"
            "# 原始片段\n"
            f"{rag_context}\n\n"
            f"# 问题\n{raw_question}\n"
            "# 输出格式\n"
            "观察员: ...\n时间线: ...\n推理官: 结论 + 概率判断 + 下一步建议\n"
        )

        try:
            func_mgr = self.context.get_llm_tool_manager()
            yield event.request_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                func_tool_manager=func_mgr,
                contexts=[],
            )
        except Exception as exc:
            logger.error(f"LLM 请求失败: {exc}")
            yield event.plain_result(f"LLM 请求失败：{exc}")

    # ---------------- internal helpers ---------------- #

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _append_session_message(
        self,
        session_id: str,
        msg: SessionMessage,
        prepend: bool = False,
    ) -> bool:
        key = self._hash_message(msg)
        hash_set = self.session_hash_sets[session_id]
        if key in hash_set:
            return False

        history = self.session_histories[session_id]
        seq = self.session_hash_sequences[session_id]

        if prepend:
            history.appendleft(msg)
            seq.appendleft(key)
        else:
            history.append(msg)
            seq.append(key)
            if len(history) > self.session_history_limit:
                history.popleft()
                removed_key = seq.popleft()
                hash_set.discard(removed_key)
        hash_set.add(key)
        return True

    def _rebuild_indexes(self, session_id: str, msg: SessionMessage):
        """根据加载的消息重建哈希索引。"""
        key = self._hash_message(msg)
        self.session_hash_sets[session_id].add(key)
        self.session_hash_sequences[session_id].append(key)

    def _hash_message(self, msg: SessionMessage) -> str:
        ts = int(msg.timestamp.timestamp())
        content_key = msg.content[:120]
        return f"{ts}|{msg.sender_id}|{content_key}"

    def _record_target_message(self, state: TargetState, msg: SessionMessage):
        state.records.append(msg)
        while len(state.records) > self.target_history_limit:
            state.records.popleft()
        state.last_updated = msg.timestamp
        asyncio.create_task(self._evaluate_forecast_rules(state))

    async def _evaluate_forecast_rules(self, state: TargetState):
        """
        评估并触发与目标相关的展望未来规则。

        Args:
            state: 目标的状态，包含其最近的消息记录。
        """
        session_rules = self.forecast_rules.get(state.session_id)
        if not session_rules:
            return
        now = self._now()
        for rule in session_rules.values():
            if rule.target_id != state.target_id or not rule.keywords:
                continue
            recent = self._recent_records(state.records, rule.window_hours, now)
            if not recent:
                continue
            likelihood = self._compute_rule_likelihood(recent, rule.keywords)
            if likelihood >= rule.threshold and self._forecast_cooldown_ok(rule, now):
                rule.last_triggered = now
                await self._save_forecast_rules()
                asyncio.create_task(
                    self._dispatch_forecast_alert(rule, state, likelihood, recent[-5:]),
                )

    def _recent_records(
        self,
        records: deque[SessionMessage],
        window_hours: int,
        now: datetime,
    ) -> list[SessionMessage]:
        cutoff = now - timedelta(hours=max(1, window_hours))
        recent = [rec for rec in records if rec.timestamp >= cutoff]
        return recent[-40:]

    def _compute_rule_likelihood(
        self,
        records: list[SessionMessage],
        keywords: list[str],
    ) -> float:
        if not records or not keywords:
            return 0.0
        kw_lower = [kw.lower() for kw in keywords]
        hits = 0
        for rec in records:
            content_lower = rec.content.lower()
            if any(kw in content_lower for kw in kw_lower):
                hits += 1
        return hits / len(records)

    def _forecast_cooldown_ok(self, rule: ForecastRule, now: datetime) -> bool:
        if not rule.last_triggered:
            return True
        return (now - rule.last_triggered) >= timedelta(minutes=rule.cooldown_minutes)

    async def _dispatch_forecast_alert(
        self,
        rule: ForecastRule,
        state: TargetState,
        likelihood: float,
        tail_records: list[SessionMessage],
    ):
        snippets = []
        for rec in tail_records:
            snippet = rec.content
            if len(snippet) > 50:
                snippet = snippet[:50] + "..."
            snippets.append(f"{self._fmt_ts(rec.timestamp)} {snippet}")
        msg = (
            "【展望未来】预测触发\n"
            f"- 规则: {rule.label}\n"
            f"- 目标: {self.user_names.get(rule.target_id, rule.target_id)}\n"
            f"- 匹配度: {likelihood * 100:.0f}% (阈值 {rule.threshold * 100:.0f}%)\n"
            f"- 关键词: {', '.join(rule.keywords)}\n"
            f"- 取样窗口: 近 {rule.window_hours} 小时，共 {len(tail_records)} 条\n"
        )
        if snippets:
            msg += "- 参考片段:\n" + "\n".join(snippets)
        try:
            await self.context.send_message(
                state.session_id,
                MessageEventResult().message(msg),
            )
        except Exception as exc:
            logger.error(f"发送展望未来告警失败: {exc}")

    async def _load_forecast_rules(self):
        if not os.path.exists(self.forecast_store_path):
            return

        def _read_file():
            try:
                with open(self.forecast_store_path, encoding="utf-8") as fp:
                    return json.load(fp)
            except (OSError, json.JSONDecodeError) as exc:
                logger.error(f"读取展望未来规则失败: {exc}")
                return None

        data = await asyncio.to_thread(_read_file)
        if data is None:
            return

        for session_id, rules in data.items():
            session_map: dict[str, ForecastRule] = {}
            for label, payload in rules.items():
                try:
                    session_map[label] = ForecastRule.from_dict(payload)
                except (KeyError, TypeError, ValueError) as exc:
                    logger.warning(f"规则 {label} 解析失败: {exc}")
            if session_map:
                self.forecast_rules[session_id] = session_map

    async def _save_forecast_rules(self):
        serializable = {
            session_id: {label: rule.to_dict() for label, rule in rules.items()}
            for session_id, rules in self.forecast_rules.items()
            if rules
        }

        def _write_file():
            try:
                os.makedirs(os.path.dirname(self.forecast_store_path), exist_ok=True)
                with open(self.forecast_store_path, "w", encoding="utf-8") as fp:
                    json.dump(serializable, fp, ensure_ascii=False, indent=2)
            except OSError as exc:
                logger.error(f"写入展望未来规则失败: {exc}")

        await asyncio.to_thread(_write_file)

    async def _load_tracked_targets(self):
        if not os.path.exists(self.targets_store_path):
            return

        def _read_file():
            try:
                with open(self.targets_store_path, encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"Failed to load tracked targets: {e}")
                return None

        data = await asyncio.to_thread(_read_file)
        if not data:
            return

        for session_id, targets_data in data.items():
            for target_id, state_data in targets_data.items():
                try:
                    self.tracked_targets[session_id][target_id] = TargetState.from_dict(
                        state_data
                    )
                except (KeyError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse target state for {target_id} in {session_id}: {e}"
                    )

    async def _save_tracked_targets(self):
        serializable = {
            session_id: {
                target_id: state.to_dict() for target_id, state in targets.items()
            }
            for session_id, targets in self.tracked_targets.items()
        }

        def _write_file():
            try:
                with open(self.targets_store_path, "w", encoding="utf-8") as f:
                    json.dump(serializable, f, ensure_ascii=False, indent=2)
            except OSError as e:
                logger.error(f"Failed to save tracked targets: {e}")

        await asyncio.to_thread(_write_file)

    async def _load_session_histories(self):
        if not os.path.exists(self.sessions_store_path):
            return

        def _read_file():
            try:
                with open(self.sessions_store_path, encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"Failed to load session histories: {e}")
                return None

        data = await asyncio.to_thread(_read_file)
        if not data:
            return

        for session_id, messages_data in data.items():
            history = deque()
            for msg_data in messages_data:
                try:
                    history.append(SessionMessage.from_dict(msg_data))
                except (KeyError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse session message in {session_id}: {e}"
                    )
            self.session_histories[session_id] = history

    async def _save_session_histories(self):
        serializable = {
            session_id: [msg.to_dict() for msg in history]
            for session_id, history in self.session_histories.items()
        }

        def _write_file():
            try:
                with open(self.sessions_store_path, "w", encoding="utf-8") as f:
                    json.dump(serializable, f, ensure_ascii=False, indent=2)
            except OSError as e:
                logger.error(f"Failed to save session histories: {e}")

        await asyncio.to_thread(_write_file)

    async def _backfill_session_history(self, session_id: str, platform_id: str) -> int:
        """
        从平台提供的历史消息存档中回填会话历史。

        Args:
            session_id: 会话 ID。
            platform_id: 平台 ID。

        Returns:
            成功导入并追加到内存中的消息数量。
        """
        mgr = getattr(self.context, "message_history_manager", None)
        if not mgr:
            return 0
        imported = 0
        page = 1
        page_size = 200
        collected: list[SessionMessage] = []
        while imported < self.backfill_limit:
            try:
                records = await mgr.get(
                    platform_id=platform_id,
                    user_id=session_id,
                    page=page,
                    page_size=page_size,
                )
            except Exception as exc:
                logger.warning(f"读取历史消息失败: {exc}")
                break
            if not records:
                break
            for record in records:
                text = self._extract_text_from_history_content(record.content)
                if not text:
                    continue
                timestamp = record.created_at
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                msg = SessionMessage(
                    sender_id=str(record.sender_id or ""),
                    sender_name=record.sender_name or str(record.sender_id or ""),
                    content=text.strip(),
                    timestamp=timestamp,
                    channel_id=session_id,
                )
                collected.append(msg)
                imported += 1
                if imported >= self.backfill_limit:
                    break
            if len(records) < page_size:
                break
            page += 1
        collected.sort(key=lambda m: m.timestamp)
        appended = 0
        for msg in collected:
            if self._append_session_message(session_id, msg, prepend=True):
                appended += 1
        return appended

    def _extract_text_from_history_content(self, content: Any) -> str:
        """从复杂的消息历史记录内容中安全地提取纯文本。"""
        if not content:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            if isinstance(content.get("message"), str):
                return content["message"]
            if isinstance(content.get("text"), str):
                return content["text"]
            if isinstance(content.get("chain"), list):
                return self._extract_text_from_history_content(content["chain"])
            return ""  # 如果字典格式不匹配，返回空字符串

        if isinstance(content, list):
            parts = []
            for item in content:
                try:
                    if isinstance(item, dict):
                        if item.get("type") == "Plain" and "text" in item:
                            parts.append(str(item["text"]))
                        elif "message" in item:
                            parts.append(str(item["message"]))
                    elif isinstance(item, str):
                        parts.append(item)
                except (TypeError, KeyError):
                    # 忽略无法解析的片段
                    continue
            return " ".join(filter(None, parts))
        return ""

    def _reindex_target(self, state: TargetState) -> int:
        history = self.session_histories.get(state.session_id, deque())
        state.records.clear()
        for msg in history:
            if msg.sender_id == state.target_id:
                state.records.append(msg)
        if state.records:
            state.last_updated = state.records[-1].timestamp
        return len(state.records)

    def _resolve_target_id(
        self,
        event: AstrMessageEvent,
        token: str,
        require_explicit: bool,
    ) -> str | None:
        mention = self._extract_mention(event)
        if mention:
            return mention
        cleaned = self._clean_target_token(token)
        if cleaned:
            return cleaned
        if require_explicit:
            return None
        return None

    def _clean_target_token(self, token: str) -> str:
        token = (token or "").strip()
        if not token:
            return ""
        if token.lower().startswith("id:"):
            return token.split(":", 1)[1].strip()
        return token

    def _extract_mention(self, event: AstrMessageEvent) -> str | None:
        try:
            for comp in event.get_messages() or []:
                if isinstance(comp, Comp.At):
                    target = str(comp.qq)
                    if target != str(event.get_self_id()):
                        return target
        except Exception:
            pass
        return None

    def _get_target_state(self, session_id: str, target_id: str) -> TargetState | None:
        return self.tracked_targets.get(session_id, {}).get(target_id)

    def _build_snapshot(self, records: list[SessionMessage], window_hours: int) -> str:
        if not records:
            return "记录为空，无法生成快照。"

        count = len(records)
        first = records[0].timestamp
        last = records[-1].timestamp
        sender_name = records[0].sender_name or "未知用户"

        question_count = sum(
            1
            for msg in records
            if msg.content.endswith("?") or msg.content.endswith("？")
        )
        pos_hits = self._count_hits(records, self.positive_keywords)
        neg_hits = self._count_hits(records, self.negative_keywords)
        topics = self._count_hits(records, self.activity_keywords)

        lines = [
            f"【{sender_name}】最近 {window_hours} 小时快照",
            f"- 发言：{count} 条（{self._fmt_ts(first)} ~ {self._fmt_ts(last)}）",
            f"- 提问密度：{question_count} 条 / {count} 条",
            f"- 情绪：{self._describe_emotion(pos_hits, neg_hits)}",
            f"- 动作信号：{self._format_hits(topics)}",
        ]

        last_content = records[-1].content
        if last_content:
            last_excerpt = last_content
            if len(last_excerpt) > 60:
                last_excerpt = last_excerpt[:60] + "..."
            lines.append(f"- 最新一句：{last_excerpt}")

        return "\n".join(lines)

    def _count_hits(self, records: Iterable[SessionMessage], keywords: list[str]):
        counter = {}
        for msg in records:
            for kw in keywords:
                if kw in msg.content:
                    counter[kw] = counter.get(kw, 0) + 1
        return counter

    def _describe_emotion(self, positives: dict, negatives: dict) -> str:
        pos_total = sum(positives.values())
        neg_total = sum(negatives.values())
        if not pos_total and not neg_total:
            base = "语气平稳"
        elif pos_total >= neg_total + 1:
            base = "整体偏积极"
        elif neg_total >= pos_total + 1:
            base = "压力较大"
        else:
            base = "情绪波动"
        detail_parts = []
        if positives:
            detail_parts.append("+ " + self._format_hits(positives, fallback=""))
        if negatives:
            detail_parts.append("- " + self._format_hits(negatives, fallback=""))
        detail = "; ".join(part for part in detail_parts if part.strip())
        return f"{base}{(' | ' + detail) if detail else ''}"

    def _format_hits(
        self,
        counter: dict[str, int],
        fallback: str = "无显著模式",
    ) -> str:
        if not counter:
            return fallback
        top = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:3]
        return ", ".join(f"{k}×{v}" for k, v in top)

    async def _query_history_db(
        self,
        session_id: str,
        target_id: str,
        start: datetime,
        end: datetime,
        keyword: str,
        limit: int = 200,
    ) -> list[dict]:
        if not os.path.exists(self.history_db_path):
            return []

        def _db_operation():
            try:
                with sqlite3.connect(self.history_db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        """
                        SELECT message_text, created_at
                        FROM messages
                        WHERE session_id = ?
                          AND sender_id = ?
                          AND created_at >= ?
                          AND created_at < ?
                          AND lower(message_text) LIKE ?
                        ORDER BY created_at ASC
                        LIMIT ?
                        """,
                        (
                            session_id,
                            target_id,
                            int(start.timestamp()),
                            int(end.timestamp()),
                            f"%{keyword.lower()}%",
                            limit,
                        ),
                    )
                    return cursor.fetchall()
            except sqlite3.Error as exc:
                logger.error(f"无法读取历史索引库: {exc}")
                return []

        rows = await asyncio.to_thread(_db_operation)
        return [
            {
                "timestamp": datetime.fromtimestamp(row["created_at"], tz=timezone.utc),
                "content": row["message_text"],
            }
            for row in rows
        ]

    def _query_memory_for_day(
        self,
        records: Iterable[SessionMessage],
        start: datetime,
        end: datetime,
        keyword: str,
    ) -> list[dict]:
        keyword_lower = keyword.lower()
        results = []
        for rec in records:
            if start <= rec.timestamp <= end and keyword_lower in rec.content.lower():
                results.append({"timestamp": rec.timestamp, "content": rec.content})
        return results

    def _infer_time_focus(
        self,
        question: str,
    ) -> tuple[datetime, datetime | None, str] | None:
        """
        从问题中推断时间焦点（如“今天”、“昨天”）。

        Args:
            question: 用户提出的原始问题。

        Returns:
            一个元组，包含时间范围的开始、结束（可选）和标签，如果未检测到则返回 None。
        """
        now = self._now().astimezone()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        tokens = question.lower()
        if "今天" in question or "today" in tokens:
            return today_start, now, "今天"
        if "今晚" in question or "tonight" in tokens:
            return today_start.replace(hour=18), now, "今晚"
        if "昨天" in question or "yesterday" in tokens:
            start = today_start - timedelta(days=1)
            end = today_start
            return start, end, "昨天"
        if "前天" in question:
            start = today_start - timedelta(days=2)
            end = today_start - timedelta(days=1)
            return start, end, "前天"
        if "上周" in question or "last week" in tokens:
            start = today_start - timedelta(days=7)
            return start, today_start, "上周"
        return None

    def _build_rag_chunks(
        self,
        session_id: str,
        target_id: str,
        question: str,
        time_focus: tuple[datetime, datetime | None, str] | None,
    ) -> list[dict]:
        """
        构建用于 RAG（检索增强生成）的上下文片段。

        通过对会话历史进行评分和排序，选取与问题最相关的消息片段。

        Args:
            session_id: 当前会话 ID。
            target_id: 目标用户 ID。
            question: 用户提出的问题。
            time_focus: 从问题中推断出的时间焦点。

        Returns:
            一个按相关性排序的字典列表，每个字典包含分数、时间戳和消息片段。
        """
        history = list(self.session_histories.get(session_id, []))
        if not history:
            return []
        tokens = self._tokenize(question)
        focus_center = (
            time_focus[1] - (time_focus[1] - time_focus[0]) / 2
            if time_focus and time_focus[1]
            else (time_focus[0] if time_focus else self._now())
        )
        scored = []
        for idx, msg in enumerate(history):
            if msg.sender_id != target_id:
                continue
            score = self._score_message(msg, tokens, focus_center, time_focus)
            if score <= 0:
                continue
            snippet_records = self._collect_context_window(
                history, idx, self.context_span
            )
            snippet = "\n".join(
                f"{self._fmt_ts(rec.timestamp)} {rec.sender_name}: {rec.content}"
                for rec in snippet_records
            )
            scored.append(
                {
                    "score": score,
                    "timestamp": self._fmt_ts(msg.timestamp),
                    "snippet": snippet,
                },
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: self.rag_context_limit]

    def _tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        current = []
        for ch in text:
            if self._is_token_char(ch):
                current.append(ch.lower())
            else:
                if current:
                    tokens.append("".join(current))
                    current = []
        if current:
            tokens.append("".join(current))
        bigrams = {
            text[i : i + 2]
            for i in range(len(text) - 1)
            if self._is_cjk(text[i]) and self._is_cjk(text[i + 1])
        }
        return tokens + list(bigrams)

    def _is_token_char(self, ch: str) -> bool:
        return ch.isalnum() or self._is_cjk(ch)

    def _is_cjk(self, ch: str) -> bool:
        return "\u4e00" <= ch <= "\u9fff"

    def _score_message(
        self,
        msg: SessionMessage,
        tokens: list[str],
        focus_center: datetime,
        time_focus: tuple[datetime, datetime | None, str] | None,
    ) -> float:
        """
        根据与问题的相关性为单条消息评分。

        评分综合考虑了关键词匹配度和时间接近度。

        Args:
            msg: 待评分的消息。
            tokens: 从问题中提取的关键词和二元组。
            focus_center: 时间焦点的中心点。
            time_focus: 推断出的时间范围。

        Returns:
            消息的相关性分数。
        """
        if not msg.content:
            return 0.0
        token_hits = sum(1 for tk in tokens if tk and tk in msg.content)
        if token_hits == 0:
            # 兜底：意图关键词
            token_hits = sum(1 for kw in self.activity_keywords if kw in msg.content)
        if token_hits == 0:
            return 0.0
        diff_hours = abs((msg.timestamp - focus_center).total_seconds()) / 3600
        time_weight = 1 / (1 + diff_hours / 6)
        if time_focus:
            start, end, _ = time_focus
            if msg.timestamp < start:
                time_weight *= 0.4
            elif end and msg.timestamp > end:
                time_weight *= 0.7
            else:
                time_weight *= 1.3
        return token_hits * 1.2 + time_weight

    def _collect_context_window(
        self,
        history: list[SessionMessage],
        idx: int,
        span: int,
    ) -> list[SessionMessage]:
        start = max(0, idx - span)
        end = min(len(history), idx + span + 1)
        return history[start:end]

    def _describe_coverage(self, state: TargetState) -> str:
        if not state or not state.records:
            return "尚无个人记录"
        try:
            first = state.records[0].timestamp
            last = state.records[-1].timestamp
            return (
                f"{len(state.records)} 条，时间跨度 {self._fmt_ts(first)} ~ {self._fmt_ts(last)}，"
                f"最近 {self._fmt_delta(self._now() - last)}更新"
            )
        except IndexError:
            return "记录为空，无法描述覆盖范围。"

    def _format_time_focus(
        self,
        time_focus: tuple[datetime, datetime | None, str] | None,
    ) -> str:
        if not time_focus:
            return "未检测到特定时间约束，默认优先匹配最近 72 小时。"
        start, end, label = time_focus
        if end:
            return f"{label}（{start.strftime('%m-%d %H:%M')} ~ {end.strftime('%m-%d %H:%M')}）"
        return f"{label} 起，自 {start.strftime('%m-%d %H:%M')} 之后的记录"

    def _fmt_ts(self, ts: datetime) -> str:
        return ts.astimezone().strftime("%m-%d %H:%M")

    def _fmt_delta(self, delta: timedelta) -> str:
        minutes = int(delta.total_seconds() // 60)
        if minutes < 1:
            return "刚刚"
        if minutes < 60:
            return f"{minutes} 分钟前"
        hours = minutes // 60
        if hours < 24:
            return f"{hours} 小时前"
        days = hours // 24
        return f"{days} 天前"
