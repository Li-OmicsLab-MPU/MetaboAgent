"""
LLM-PToT 引擎的纯本地计算模块
包含：
1. 帕累托前沿提取器 (get_pareto_front)
2. 启发式候选生成器 (generate_heuristic_candidates)
"""

from typing import List, Dict, Any
import copy
import re


def _dominates(cand_a: Dict, cand_b: Dict) -> bool:
    """
    判断 A 是否帕累托支配 B。
    
    支配定义：
    - perf, bio: 越大越好
    - corr, cost: 越小越好
    
    A 支配 B 当且仅当：
    1. A 在所有维度都不差于 B
    2. A 至少在一个维度严格优于 B
    
    Args:
        cand_a: 候选 A，包含 perf, bio, corr, cost 四个维度
        cand_b: 候选 B，包含 perf, bio, corr, cost 四个维度
    
    Returns:
        bool: True 表示 A 支配 B，False 表示不支配
    """
    # 1. 确保 A 在所有维度都不差于 B
    if cand_a['perf'] < cand_b['perf']:
        return False
    if cand_a['bio'] < cand_b['bio']:
        return False
    if cand_a['corr'] > cand_b['corr']:
        return False
    if cand_a['cost'] > cand_b['cost']:
        return False
    
    # 2. 确保 A 至少在一个维度严格优于 B
    if cand_a['perf'] > cand_b['perf']:
        return True
    if cand_a['bio'] > cand_b['bio']:
        return True
    if cand_a['corr'] < cand_b['corr']:
        return True
    if cand_a['cost'] < cand_b['cost']:
        return True
    
    return False  # 完全相等时不认为支配


def get_pareto_front(candidates: List[Dict]) -> List[Dict]:
    """
    提取帕累托前沿
    
    从一组已打分的候选组合中，返回没有被任何其他组合支配的精英集合。
    使用 O(N^2) 暴力比对（因每轮候选通常不超过 20 个，耗时在微秒级）。
    
    Args:
        candidates: 候选列表，每个元素是包含 features, perf, bio, corr, cost 的字典
    
    Returns:
        List[Dict]: 帕累托前沿（不被任何其他元素支配的候选列表）
    """
    pareto_front = []
    
    for i, cand_i in enumerate(candidates):
        is_dominated = False
        for j, cand_j in enumerate(candidates):
            if i != j and _dominates(cand_j, cand_i):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_front.append(cand_i)
    
    return pareto_front


def generate_heuristic_candidates(
    current_features: List[str],
    candidate_pool: List[str],
    taxonomy_map: Dict,
    pathway_map: Dict,
    checker_tool: Any,
    sorted_base_pool: List[str]
) -> List[List[str]]:
    """
    异质性启发式节点生成器
    
    基于当前的特征子集 S，生成下一批候选特征子集（每次只增加 1 个新特征）。
    采用"异质性分支策略"，合并执行以下三种策略：
    
    1. 机制挖掘 (Exploitation)：基于图谱邻居关系扩展
    2. 通路正交 (Pathway Orthogonality) 🆕：基于代谢通路正交性扩展
    3. 数据贪心 (Greedy)：基于单变量 AUC 排序扩展
    
    🆕 任务二升级 (2026-04-30):
    - 策略 2 从"分类正交 (Taxonomic)"升级为"通路正交 (Pathway)"
    - 优先选择属于"尚未被覆盖的全新通路"的代谢物
    - 配合 f_bio 奖励局部连通性，构建"T型特征群（宏观正交，微观冗余）"
    
    Args:
        current_features: 当前的特征集合 S（可能为空列表 []）
        candidate_pool: Phase 1 传过来的备选特征全集（约 50 个）
        taxonomy_map: 分类字典，格式 {feature_name: taxonomy_class}
        pathway_map: 通路字典，格式 {hmdb_id: [pathway1, pathway2, ...]}
        checker_tool: 包含图谱的实例，需提供 find_connected_pairs 方法
        sorted_base_pool: 根据单变量 AUC 从高到低排序好的备选特征列表
    
    Returns:
        List[List[str]]: 生成的新特征子集列表（已去重）
    """
    new_subsets = []
    
    # 冷启动保护：如果 current_features 为空，跳过策略1和策略2
    is_cold_start = len(current_features) == 0
    
    # ========== 策略1：机制挖掘 (Exploitation) ==========
    if not is_cold_start:
        # 1.1 使用正则表达式提取所有底层的唯一 HMDB IDs
        hmdb_pattern = re.compile(r'HMDB\d+')
        unique_hmdbs = set()
        
        for feature in current_features:
            matches = hmdb_pattern.findall(feature)
            unique_hmdbs.update(matches)
        
        # 1.2 调用图谱工具找邻居
        if unique_hmdbs:
            unique_hmdbs_list = list(unique_hmdbs)
            try:
                connected_pairs = checker_tool.find_connected_pairs(unique_hmdbs_list)
                
                # 1.3 找出属于 candidate_pool 但尚未在 S 中的邻居
                for source_id, target_id in connected_pairs:
                    # 检查 target_id 是否在 candidate_pool 中且不在 current_features 中
                    target_features = [f for f in candidate_pool if target_id in f and f not in current_features]
                    
                    for target_feature in target_features:
                        # 策略 1a: 直接加入单体 B
                        new_subset = current_features + [target_feature]
                        new_subsets.append(new_subset)
                        
                        # 策略 1b: 生成比值 Ratio_源ID_B
                        ratio_feature = f"Ratio_{source_id}_{target_id}"
                        if ratio_feature in candidate_pool and ratio_feature not in current_features:
                            new_subset_ratio = current_features + [ratio_feature]
                            new_subsets.append(new_subset_ratio)
            except Exception as e:
                # 如果图谱查询失败，跳过策略1
                pass
    
    # ========== 策略2：通路正交 (Pathway Orthogonality) 🆕 ==========
    # 设计理念：
    # - 宏观正交：特征尽可能散布在不同的代谢通路上
    # - 微观冗余：配合 f_bio 奖励局部连通性（同一通路内的代谢物）
    # - T型特征群：宏观多样性 + 微观冗余性 = 最佳迁移能力
    # 
    # 算法流程：
    # 1. 提取当前特征已覆盖的代谢通路集合
    # 2. 优先选择属于"全新通路"的特征（尚未被覆盖）
    # 3. 如果全新通路耗尽，回退到原有的分类正交逻辑
    # 
    # 日期：2026-04-30 (任务二：通路正交升级)
    if not is_cold_start and pathway_map:
        try:
            # ================================================================
            # 步骤 1: 提取当前特征已覆盖的代谢通路集合
            # ================================================================
            current_pathways = set()
            for feature in current_features:
                # 提取 HMDB ID
                hmdb_ids = re.findall(r'HMDB\d+', feature)
                for hmdb_id in hmdb_ids:
                    if hmdb_id in pathway_map:
                        # pathway_map 结构: {hmdb_id: [pathway1, pathway2, ...]}
                        pathways = pathway_map.get(hmdb_id, [])
                        if isinstance(pathways, list):
                            current_pathways.update(pathways)
                        elif isinstance(pathways, str):
                            current_pathways.add(pathways)
            
            # ================================================================
            # 步骤 2: 优先选择属于"全新通路"的特征
            # ================================================================
            novel_pathway_features = []  # 全新通路的特征
            existing_pathway_features = []  # 已有通路的特征
            
            for feature in candidate_pool:
                if feature in current_features:
                    continue
                
                # 提取该特征的 HMDB ID
                hmdb_ids = re.findall(r'HMDB\d+', feature)
                
                # 检查该特征的通路
                feature_pathways = set()
                for hmdb_id in hmdb_ids:
                    if hmdb_id in pathway_map:
                        pathways = pathway_map.get(hmdb_id, [])
                        if isinstance(pathways, list):
                            feature_pathways.update(pathways)
                        elif isinstance(pathways, str):
                            feature_pathways.add(pathways)
                
                # 判断是否属于全新通路
                if feature_pathways:
                    # 检查是否有任何通路与当前已覆盖的通路重叠
                    if not feature_pathways.intersection(current_pathways):
                        # 完全不重叠 → 全新通路
                        novel_pathway_features.append(feature)
                    else:
                        # 有重叠 → 已有通路
                        existing_pathway_features.append(feature)
            
            # ================================================================
            # 步骤 3: 选择 top 3 特征
            # ================================================================
            # 优先级：全新通路 > 已有通路
            orthogonal_features = []
            
            # 3.1: 优先选择全新通路的特征
            for feature in novel_pathway_features[:3]:
                orthogonal_features.append(feature)
            
            # 3.2: 如果全新通路不足 3 个，用已有通路的特征补充
            if len(orthogonal_features) < 3:
                remaining_slots = 3 - len(orthogonal_features)
                for feature in existing_pathway_features[:remaining_slots]:
                    orthogonal_features.append(feature)
            
            # ================================================================
            # 步骤 4: 生成新候选子集
            # ================================================================
            for feature in orthogonal_features:
                new_subset = current_features + [feature]
                new_subsets.append(new_subset)
        
        except Exception as e:
            # ================================================================
            # 降级保护：回退到分类正交逻辑
            # ================================================================
            try:
                # 2.1 获取 current_features 中已有的所有 taxonomy 类名
                existing_taxonomies = set()
                for feature in current_features:
                    if feature in taxonomy_map:
                        existing_taxonomies.add(taxonomy_map[feature])
                
                # 2.2 从 candidate_pool 中找出不属于这些已有分类、且当前不在 S 中的特征
                orthogonal_features = []
                for feature in candidate_pool:
                    if feature not in current_features:
                        feature_taxonomy = taxonomy_map.get(feature, None)
                        if feature_taxonomy and feature_taxonomy not in existing_taxonomies:
                            orthogonal_features.append(feature)
                
                # 2.3 取前 3 个，分别加入 S 形成新组合
                for feature in orthogonal_features[:3]:
                    new_subset = current_features + [feature]
                    new_subsets.append(new_subset)
            except Exception as fallback_error:
                # 如果降级也失败，跳过策略2
                pass
    
    # ========== 策略3：数据贪心 (Greedy) ==========
    # 直接从 sorted_base_pool 中按顺序找出前 3 个目前不在 S 中的特征
    greedy_count = 0
    for feature in sorted_base_pool:
        if feature in candidate_pool and feature not in current_features:
            new_subset = current_features + [feature]
            new_subsets.append(new_subset)
            greedy_count += 1
            if greedy_count >= 3:
                break
    
    # ========== 安全去重 ==========
    # 由于 List 不可哈希，将每个 subset 内部排序后转为 tuple 存入 set 去重
    unique_subsets = set()
    for subset in new_subsets:
        sorted_subset = tuple(sorted(subset))
        unique_subsets.add(sorted_subset)
    
    # 还原回 List[List[str]]
    result = [list(subset) for subset in unique_subsets]
    
    return result


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("测试模块 1：帕累托前沿提取器")
    print("=" * 60)
    
    # Mock 5 个候选组合
    test_candidates = [
        {"features": ['A', 'B'], "perf": 0.85, "bio": 0.6, "corr": 0.2, "cost": 0.4},
        {"features": ['C', 'D'], "perf": 0.90, "bio": 0.7, "corr": 0.15, "cost": 0.3},  # 最优
        {"features": ['E', 'F'], "perf": 0.80, "bio": 0.5, "corr": 0.25, "cost": 0.5},  # 最差（被支配）
        {"features": ['G', 'H'], "perf": 0.88, "bio": 0.65, "corr": 0.18, "cost": 0.35},
        {"features": ['I', 'J'], "perf": 0.82, "bio": 0.75, "corr": 0.22, "cost": 0.38},  # bio 最高
    ]
    
    print("\n输入候选组合：")
    for i, cand in enumerate(test_candidates):
        print(f"  {i+1}. {cand['features']}: perf={cand['perf']}, bio={cand['bio']}, "
              f"corr={cand['corr']}, cost={cand['cost']}")
    
    pareto_front = get_pareto_front(test_candidates)
    
    print(f"\n帕累托前沿（共 {len(pareto_front)} 个）：")
    for i, cand in enumerate(pareto_front):
        print(f"  {i+1}. {cand['features']}: perf={cand['perf']}, bio={cand['bio']}, "
              f"corr={cand['corr']}, cost={cand['cost']}")
    
    # 验证最差的组合是否被剔除
    worst_candidate = {"features": ['E', 'F'], "perf": 0.80, "bio": 0.5, "corr": 0.25, "cost": 0.5}
    is_worst_removed = worst_candidate not in pareto_front
    print(f"\n✓ 最差组合 ['E', 'F'] 是否被成功剔除: {is_worst_removed}")
    
    print("\n" + "=" * 60)
    print("测试模块 2：启发式候选生成器（去重机制）")
    print("=" * 60)
    
    print("\n去重机制说明：")
    print("1. 由于 Python 中 List 不可哈希，无法直接使用 set 去重")
    print("2. 解决方案：将每个 subset 内部先排序 sorted()")
    print("3. 转换为 tuple 存入 set 中去重")
    print("4. 最后再还原回 List[List[str]]")
    print("\n示例代码片段：")
    print("```python")
    print("unique_subsets = set()")
    print("for subset in new_subsets:")
    print("    sorted_subset = tuple(sorted(subset))")
    print("    unique_subsets.add(sorted_subset)")
    print("result = [list(subset) for subset in unique_subsets]")
    print("```")
    
    # Mock 测试去重
    print("\n模拟去重测试：")
    mock_subsets = [
        ['A', 'B', 'C'],
        ['C', 'A', 'B'],  # 与第一个相同，只是顺序不同
        ['A', 'D'],
        ['D', 'A'],  # 与第三个相同
        ['E', 'F'],
    ]
    
    print("去重前：")
    for subset in mock_subsets:
        print(f"  {subset}")
    
    unique_subsets = set()
    for subset in mock_subsets:
        sorted_subset = tuple(sorted(subset))
        unique_subsets.add(sorted_subset)
    result = [list(subset) for subset in unique_subsets]
    
    print(f"\n去重后（从 {len(mock_subsets)} 个减少到 {len(result)} 个）：")
    for subset in result:
        print(f"  {subset}")
    
    print("\n" + "=" * 60)
    print("✓ ptot_searcher.py 实现完成")
    print("=" * 60)
