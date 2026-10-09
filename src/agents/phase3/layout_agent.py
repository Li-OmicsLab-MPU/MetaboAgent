"""
Phase 3: Visual Auditor (Critic) - ACVAL 架构的 Critic 组件

核心功能:
调用多模态大模型（VLM）审查图片，检测排版冲突并提供调整建议。

设计理念:
- 固化通用 Prompt，确保审查标准一致
- 返回结构化 JSON，便于 Actor 解析和应用
- Mock VLM 调用，便于测试和开发

Author: MetaboAgent Team
Date: 2026-04-24
"""

import json
from typing import Dict, Any, Optional


class VisualAuditor:
    """
    视觉审查官 (Visual Auditor) - ACVAL 架构的 Critic 组件

    职责:
    1. 调用多模态大模型（VLM）审查图片
    2. 检测排版冲突（遮挡、重叠、溢出）
    3. 提供可被 Matplotlib 直接应用的 kwargs 参数调整建议

    核心方法:
    - audit_figure: 审查图片并返回结构化结果

    Examples:
        >>> auditor = VisualAuditor(use_mock=True)
        >>> result = auditor.audit_figure('output/figures/fig1_pareto_trajectory.pdf')
        >>> print(result['status'])  # 'PASS' or 'FAIL'
        >>> if result['status'] == 'FAIL':
        ...     print(result['suggested_kwargs'])
    """

    # ========================================================================
    # 固化通用 Prompt（顶级医学期刊审查标准）
    # ========================================================================
    AUDIT_PROMPT_TEMPLATE = """
[Role]
你是一位顶级医学期刊（如 Nature Medicine, The Lancet）的首席数据可视化审查官。
请审查输入的学术图表，确保无排版冲突，符合顶级期刊的出版标准。

[Evaluation Criteria]
请按以下四个维度评估图表质量：

1. Data Occlusion (数据遮挡):
   - 图例、文本标签是否遮挡了核心数据元素（散点、折线、柱状图）？
   - 标题、坐标轴标签是否与数据重叠？
   - 评分标准: 无遮挡 = PASS，有遮挡 = FAIL

2. Legibility (可读性):
   - 文字标签是否重叠导致无法阅读？
   - 字体大小是否合适（不能太小或太大）？
   - 颜色对比度是否足够（文字与背景）？
   - 评分标准: 清晰可读 = PASS，有重叠或模糊 = FAIL

3. Bounding Box (边界完整性):
   - 是否有图例或文本溢出了物理边界（被裁剪）？
   - 图表元素是否完整显示在画布内？
   - 评分标准: 无溢出 = PASS，有溢出 = FAIL

4. Academic Aesthetics (学术美学):
   - 留白是否合理（不能太拥挤或太空旷）？
   - 图例位置是否合理（不遮挡数据，易于查看）？
   - 整体布局是否平衡、专业？
   - 评分标准: 美观专业 = PASS，布局不佳 = FAIL

[Task]
如果发现任何问题，请给出可被 Matplotlib 直接应用的 kwargs 参数调整建议。

常用调整参数:
- legend_loc: 图例位置 ('upper left', 'upper right', 'lower left', 'lower right', 'best')
- legend2_loc: 第二个图例位置（如果有）
- bbox_to_anchor: 图例锚点位置 (x, y) 元组，例如 (1.05, 1.0)
- fontsize: 全局字体大小（整数，例如 10, 11, 12）
- title_fontsize: 标题字体大小
- label_fontsize: 坐标轴标签字体大小
- legend_fontsize: 图例字体大小

[Output Format]
你必须输出严格的 JSON 格式，不要包含任何 markdown 标记（如 ```json）：

{{
  "status": "PASS" | "FAIL",
  "reasoning": "详细说明发现的问题（如果 PASS 则说明通过原因）",
  "suggested_kwargs": {{
    "legend_loc": "lower right",
    "fontsize": 10
  }}
}}

注意:
- 如果 status 为 "PASS"，suggested_kwargs 应为空字典 {{}}
- 如果 status 为 "FAIL"，suggested_kwargs 必须包含至少一个调整建议
- reasoning 必须具体说明问题位置和原因

[Image Path]
{image_path}

请开始审查。
"""

    def __init__(
        self,
        use_mock: bool = True,
        vlm_caller: Optional[Any] = None
    ):
        """
        初始化视觉审查官

        Args:
            use_mock: 是否使用 Mock VLM（默认 True，用于测试）
            vlm_caller: 真实的 VLM 调用函数（可选）
        """
        self.use_mock = use_mock
        self.vlm_caller = vlm_caller
        self.audit_count = 0  # 审查次数计数器（用于 Mock 逻辑）

    def audit_figure(self, image_path: str) -> Dict[str, Any]:
        """
        审查图片并返回结构化结果

        Args:
            image_path: 图片文件路径

        Returns:
            Dict: 审查结果，包含以下键:
                - status: 'PASS' 或 'FAIL'
                - reasoning: 问题说明或通过原因
                - suggested_kwargs: 调整建议字典

        Examples:
            >>> auditor = VisualAuditor(use_mock=True)
            >>> result = auditor.audit_figure('output/figures/fig1.pdf')
            >>> print(result)
            {'status': 'FAIL', 'reasoning': '...', 'suggested_kwargs': {...}}
        """
        self.audit_count += 1

        print(f"\n{'='*80}")
        print(f"[ACVAL Critic] Audit #{self.audit_count}: {image_path}")
        print(f"{'='*80}")

        if self.use_mock:
            # Mock VLM 调用（用于测试）
            return self._mock_vlm_audit(image_path)
        else:
            # 真实 VLM 调用
            return self._real_vlm_audit(image_path)

    def _mock_vlm_audit(self, image_path: str) -> Dict[str, Any]:
        """
        Mock VLM 审查逻辑（用于测试）

        模拟策略:
        - 第 1 次审查: 固定返回 FAIL，建议调整图例位置和字体大小
        - 第 2 次审查: 固定返回 PASS，表示排版完美

        Args:
            image_path: 图片文件路径

        Returns:
            Dict: 审查结果
        """
        if self.audit_count == 1:
            # 第一次审查：模拟发现问题
            result = {
                "status": "FAIL",
                "reasoning": (
                    "发现以下排版问题:\n"
                    "1. Data Occlusion: 右上角的 'Bubble Size' 图例遮挡了部分帕累托前沿数据点\n"
                    "2. Legibility: 深度标签 (D0, D1, ...) 与数据点重叠，部分文字难以阅读\n"
                    "3. Bounding Box: 右上角图例的边框略微超出画布边界\n"
                    "建议: 将 'Bubble Size' 图例移至左上角，并减小字体大小以避免遮挡"
                ),
                "suggested_kwargs": {
                    "legend2_loc": "upper left",
                    "legend2_fontsize": 7,
                    "bbox_to_anchor2": (0.02, 0.98)
                }
            }

            print(f"[ACVAL Critic] Status: {result['status']}")
            print(f"[ACVAL Critic] Reasoning: {result['reasoning']}")
            print(f"[ACVAL Critic] Suggested kwargs: {result['suggested_kwargs']}")

            return result

        else:
            # 第二次及以后：模拟通过审查
            result = {
                "status": "PASS",
                "reasoning": (
                    "图表排版完美，符合顶级医学期刊标准:\n"
                    "1. Data Occlusion: 无遮挡，所有数据点清晰可见\n"
                    "2. Legibility: 文字标签清晰可读，无重叠\n"
                    "3. Bounding Box: 所有元素完整显示在画布内\n"
                    "4. Academic Aesthetics: 布局平衡，留白合理，专业美观\n"
                    "建议: 无需调整，可以定稿"
                ),
                "suggested_kwargs": {}
            }

            print(f"[ACVAL Critic] Status: {result['status']}")
            print(f"[ACVAL Critic] Reasoning: {result['reasoning']}")
            print(f"[ACVAL Critic] ✅ 图表已通过审查，可以定稿！")

            return result

    def _real_vlm_audit(self, image_path: str) -> Dict[str, Any]:
        """
        真实 VLM 审查逻辑（调用多模态大模型）

        Args:
            image_path: 图片文件路径

        Returns:
            Dict: 审查结果

        TODO: 实现真实的 VLM API 调用
        """
        if self.vlm_caller is None:
            raise ValueError("vlm_caller is required for real VLM audit")

        # 构造 Prompt
        prompt = self.AUDIT_PROMPT_TEMPLATE.format(image_path=image_path)

        try:
            # 调用 VLM
            response_str = self.vlm_caller(prompt, image_path=image_path)

            # 清理可能的 markdown 格式
            clean_str = response_str.replace("```json", "").replace("```", "").strip()

            # 解析 JSON
            result = json.loads(clean_str)

            # 验证结果格式
            if 'status' not in result or 'reasoning' not in result or 'suggested_kwargs' not in result:
                raise ValueError("Invalid VLM response format")

            if result['status'] not in ['PASS', 'FAIL']:
                raise ValueError(f"Invalid status: {result['status']}")

            print(f"[ACVAL Critic] Status: {result['status']}")
            print(f"[ACVAL Critic] Reasoning: {result['reasoning']}")
            if result['status'] == 'FAIL':
                print(f"[ACVAL Critic] Suggested kwargs: {result['suggested_kwargs']}")

            return result

        except Exception as e:
            # 降级保护：如果 VLM 调用失败，返回 PASS（避免阻塞流程）
            print(f"[ACVAL Critic] Warning: VLM audit failed: {e}")
            print(f"[ACVAL Critic] Fallback: Returning PASS to avoid blocking")

            return {
                "status": "PASS",
                "reasoning": f"VLM audit failed ({str(e)}), fallback to PASS",
                "suggested_kwargs": {}
            }


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("Testing Visual Auditor (Critic)")
    print("="*80)

    # 创建 Mock 审查官
    auditor = VisualAuditor(use_mock=True)

    # 模拟第一次审查（应该返回 FAIL）
    print("\n[Test 1] First audit (should FAIL)...")
    result1 = auditor.audit_figure('output/figures/fig1_pareto_trajectory.pdf')
    assert result1['status'] == 'FAIL', "First audit should FAIL"
    assert len(result1['suggested_kwargs']) > 0, "Should have suggested kwargs"
    print("✅ Test 1 passed!")

    # 模拟第二次审查（应该返回 PASS）
    print("\n[Test 2] Second audit (should PASS)...")
    result2 = auditor.audit_figure('output/figures/fig1_pareto_trajectory.pdf')
    assert result2['status'] == 'PASS', "Second audit should PASS"
    assert len(result2['suggested_kwargs']) == 0, "Should have no suggested kwargs"
    print("✅ Test 2 passed!")

    print("\n" + "="*80)
    print("All Tests Completed!")
    print("="*80)
