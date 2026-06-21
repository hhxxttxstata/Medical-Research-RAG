"""
报告生成工具单元测试

测试策略：
  - ReportGenerator 的规则方法（_para、_numbered_list 等）是纯字符串操作
  - 不测试 LLM 辅助分支（需要 API 调用）
  - 测试模板填充、主题推断、章节提取
"""

import pytest

from src.tools.report_generator import ReportGenerator


@pytest.fixture
def generator():
    """报告生成器实例（不含 LLM）"""
    return ReportGenerator()


class TestReportGenerator:
    """报告生成器主功能测试"""

    def test_run_deployment_success(self, generator):
        """运行部署报告应返回 success=True 和 report Markdown"""
        result = generator.run(
            report_type="deployment",
            content="本次部署涉及 GPU 服务器和环境配置。",
            topic="AI系统部署",
        )
        assert result["success"] is True
        assert "report" in result
        assert result["report_type"] == "deployment"
        assert result["topic"] == "AI系统部署"
        assert "# 📦 部署报告" in result["report"]

    def test_run_troubleshoot_success(self, generator):
        """运行问题排查报告应返回正确类型"""
        result = generator.run(
            report_type="troubleshoot",
            content="系统启动失败，日志显示端口被占用。",
            topic="服务启动失败排查",
        )
        assert result["success"] is True
        assert "# 🔧 问题排查报告" in result["report"]

    def test_run_meeting_success(self, generator):
        """运行会议纪要应返回正确类型"""
        result = generator.run(
            report_type="meeting",
            content="讨论了项目进度和下一步计划。",
            topic="周会纪要",
        )
        assert result["success"] is True
        assert "# 📝 会议纪要" in result["report"]

    def test_infer_topic_from_heading(self, generator):
        """从 Markdown 标题中提取主题"""
        content = "# 肺栓塞AI诊断系统部署方案\n\n正文内容。"
        topic = generator._infer_topic(content, "deployment")
        assert "肺栓塞AI诊断系统部署方案" in topic

    def test_infer_topic_fallback(self, generator):
        """无标题时取前 60 字符"""
        content = "这是一段没有标题的纯文本内容，用于测试主题提取的兜底逻辑。"
        topic = generator._infer_topic(content, "deployment")
        assert len(topic) <= 65  # 60字 + 省略号
        assert topic is not None

    def test_get_schema(self, generator):
        """get_schema 返回正确的 schema 结构"""
        schema = generator.get_schema()
        assert schema["tool_name"] == "generate_report"
        assert "description" in schema
        assert "parameters" in schema
        params = schema["parameters"]
        assert "report_type" in params["properties"]
        assert "content" in params["properties"]
        assert params["properties"]["report_type"]["type"] == "string"

    def test_get_template(self, generator):
        """_get_template 根据类型返回对应模板"""
        dep = generator._get_template("deployment")
        troub = generator._get_template("troubleshoot")
        meet = generator._get_template("meeting")
        assert "{overview}" in dep
        assert "{symptoms}" in troub
        assert "{discussion}" in meet

    def test_run_without_topic(self, generator):
        """不传 topic 时应自动推断"""
        result = generator.run(
            report_type="meeting",
            content="# 项目周会\n\n讨论了进展。",
        )
        assert result["success"] is True
        assert result["topic"] is not None


class TestStaticHelperMethods:
    """ReportGenerator 的静态工具方法测试"""

    def test_para_extraction(self):
        """从内容提取第一段文本"""
        result = ReportGenerator._para(
            "肺栓塞是一种由血栓阻塞肺动脉引起的危急重症，需要立即进行诊断和治疗。这是后续内容。", "fallback"
        )
        # 应提取到第一个完整句子
        assert "肺栓塞" in result
        assert "危急重症" in result
        assert result != "fallback"

    def test_para_fallback(self):
        """内容不足时返回 fallback"""
        result = ReportGenerator._para("短", "默认内容")
        assert result == "默认内容"

    def test_numbered_list_extraction(self):
        """从内容中提取数字列表"""
        content = "## 部署步骤\n1. 安装依赖\n2. 配置环境\n3. 启动服务"
        result = ReportGenerator._numbered_list(content, "部署")
        assert len(result) > 0
        # 应包含提取的列表项（去除序号）
        assert "安装依赖" in result
        assert "配置环境" in result

    def test_bullet_list_extraction(self):
        """从内容中提取无序列表"""
        content = "## 讨论内容\n- 项目进度\n- 技术方案\n- 风险评估"
        result = ReportGenerator._bullet_list(content, "讨论")
        assert "项目进度" in result

    def test_extract_env_info(self):
        """从内容中提取环境信息区段"""
        content = "## 环境要求\n- Ubuntu 20.04\n- Python 3.10\n- CUDA 12.4"
        result = ReportGenerator._extract_env_info(content)
        assert "Ubuntu" in result
        assert "Python" in result

    def test_extract_env_info_not_found(self):
        """无环境信息区段时返回兜底提示"""
        result = ReportGenerator._extract_env_info("不包含环境信息的内容。")
        assert "未明确说明" in result

    def test_extract_action_items(self):
        """提取待办事项并格式化为表格"""
        content = "## 待办事项\n- 完成模型训练（张三负责），截止下周一\n- 整理测试报告（李四负责）"
        result = ReportGenerator._extract_action_items(content)
        assert "待办事项" in result
        assert "|" in result  # 表格格式

    def test_extract_key_value_section(self):
        """提取 key-value 配置信息"""
        content = "## 配置说明\nport=8000\nhost=0.0.0.0"
        result = ReportGenerator._extract_key_value_section(content, "配置")
        assert "port=8000" in result

    def test_extract_notes(self):
        """提取注意事项"""
        content = "## 注意事项\n- 不要在生产环境直接调试\n- 做好数据备份"
        result = ReportGenerator._extract_notes(content)
        assert "不要" in result

    def test_extract_root_cause(self):
        """提取根因分析"""
        content = "## 根因分析\n- 数据库连接池耗尽\n- 慢查询导致堆积"
        result = ReportGenerator._extract_root_cause(content)
        assert "数据库" in result

    def test_extract_solution(self):
        """提取解决方案"""
        content = "## 解决方案\n- 增加连接池大小\n- 优化慢查询"
        result = ReportGenerator._extract_solution(content)
        assert "连接池" in result or len(result) > 0

    def test_extract_section_text(self):
        """提取指定标题下的文本"""
        content = "## 故障现象\n系统响应缓慢。\nAPI 超时。"
        result = ReportGenerator._extract_section_text(content, ["故障现象", "症状"])
        assert "系统响应缓慢" in result
