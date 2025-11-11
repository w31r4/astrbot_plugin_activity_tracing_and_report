import asyncio
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from astrbot_plugin_activity_tracing_and_report.main import (
    ActivityTracingPlugin,
    SessionMessage,
    TargetState,
)


class TestActivityTracing(unittest.TestCase):
    def setUp(self):
        """为每个测试用例设置一个临时的目录和模拟对象。"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mock_context = MagicMock()

        # 模拟 get_astrbot_data_path 以使用临时目录
        self.data_path_patcher = patch(
            "astrbot_plugin_activity_tracing_and_report.main.get_astrbot_data_path",
            return_value=self.temp_dir.name,
        )
        self.mock_get_data_path = self.data_path_patcher.start()

        # 实例化插件
        self.plugin = ActivityTracingPlugin(self.mock_context)

    def tearDown(self):
        """清理临时目录和补丁。"""
        self.data_path_patcher.stop()
        self.temp_dir.cleanup()

    def test_data_model_serialization(self):
        """测试 SessionMessage 和 TargetState 的序列化和反序列化。"""
        # 测试 SessionMessage
        now = datetime.now(timezone.utc)
        msg = SessionMessage(
            sender_id="123",
            sender_name="test_user",
            content="hello",
            timestamp=now,
            channel_id="456",
        )
        msg_dict = msg.to_dict()
        self.assertEqual(msg_dict["sender_id"], "123")
        self.assertEqual(msg_dict["timestamp"], now.isoformat())
        rebuilt_msg = SessionMessage.from_dict(msg_dict)
        self.assertEqual(msg, rebuilt_msg)

        # 测试 TargetState
        state = TargetState(
            target_id="789",
            session_id="456",
            platform_id="test_platform",
            last_updated=now,
            backfilled=True,
        )
        state_dict = state.to_dict()
        self.assertEqual(state_dict["target_id"], "789")
        self.assertTrue(state_dict["backfilled"])
        rebuilt_state = TargetState.from_dict(state_dict)
        # 忽略 records 的比较，因为它不在 to_dict 的范围内
        rebuilt_state.records = state.records
        self.assertEqual(state, rebuilt_state)

    async def _test_persistence(self):
        """测试插件状态的持久化和加载。"""
        # 1. 初始化并设置一些状态
        await self.plugin.initialize()
        session_id = "session_1"
        target_id = "target_1"
        now = datetime.now(timezone.utc)

        # 添加会话历史
        msg1 = SessionMessage("u1", "n1", "c1", now, session_id)
        self.plugin.session_histories[session_id].append(msg1)

        # 添加追踪目标
        target_state = TargetState(target_id, session_id, "platform_1")
        self.plugin.tracked_targets[session_id][target_id] = target_state

        # 2. 调用 terminate 保存状态
        await self.plugin.terminate()

        # 3. 创建一个新的插件实例来加载状态
        new_plugin = ActivityTracingPlugin(self.mock_context)
        await new_plugin.initialize()

        # 4. 验证状态是否已恢复
        self.assertIn(session_id, new_plugin.session_histories)
        self.assertEqual(len(new_plugin.session_histories[session_id]), 1)
        self.assertEqual(new_plugin.session_histories[session_id][0].content, "c1")

        self.assertIn(session_id, new_plugin.tracked_targets)
        self.assertIn(target_id, new_plugin.tracked_targets[session_id])
        self.assertEqual(
            new_plugin.tracked_targets[session_id][target_id].platform_id, "platform_1"
        )

    def test_persistence_sync_wrapper(self):
        """同步运行异步的持久化测试。"""
        asyncio.run(self._test_persistence())

    def test_build_snapshot(self):
        """测试 _build_snapshot 函数能否正确生成摘要。"""
        now = datetime.now(timezone.utc)
        records = [
            SessionMessage("u1", "user1", "今天天气真好", now, "s1"),
            SessionMessage("u1", "user1", "我们去哪里吃饭？", now, "s1"),
            SessionMessage("u1", "user1", "我感到有点累", now, "s1"),
        ]
        snapshot = self.plugin._build_snapshot(records, 24)
        self.assertIn("user1", snapshot)
        self.assertIn("3 条", snapshot)
        self.assertIn("提问密度：1 条", snapshot)
        self.assertIn("情绪：", snapshot)

    def test_compute_rule_likelihood(self):
        """测试 _compute_rule_likelihood 函数能否正确计算关键词匹配度。"""
        now = datetime.now(timezone.utc)
        records = [
            SessionMessage("u1", "user1", "我们今晚去聚餐吧", now, "s1"),
            SessionMessage("u1", "user1", "顺便打游戏", now, "s1"),
            SessionMessage("u1", "user1", "然后回家", now, "s1"),
        ]
        keywords = ["聚餐", "打游戏"]
        likelihood = self.plugin._compute_rule_likelihood(records, keywords)
        self.assertAlmostEqual(likelihood, 2 / 3)


if __name__ == "__main__":
    unittest.main()
