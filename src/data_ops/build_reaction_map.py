#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KEGG 反应网络构建脚本
功能：解析 KEGG 原始数据，构建反应网络，并进行分层存储
作者：MetaboAgent Team
日期：2026-01-03
"""

import json
import re
import csv
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


def build_raw_kegg_graph(rpair_path: str, rclass_path: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Step 1: 构建原始 KEGG 反应图谱（增强版，保留 RC ID）
    
    Args:
        rpair_path: rpair 文件路径
        rclass_path: rclass 文件路径
    
    Returns:
        邻接字典 {KEGG_ID: [{"target": KEGG_ID, "rc_id": RC_ID}, ...]}
    """
    print("=" * 60)
    print("Step 1: 构建原始 KEGG 反应图谱（增强版）")
    print("=" * 60)
    
    # 1.1 解析 rpair 文件，提取 main 类型的代谢物对
    print(f"正在解析 rpair 文件: {rpair_path}")
    main_pairs = set()
    
    with open(rpair_path, 'r', encoding='utf-8') as f:
        current_name = None
        current_type = None
        
        for line in f:
            line = line.strip()
            
            if line.startswith('NAME'):
                # 提取代谢物对，例如 "NAME        C00005_C00006"
                parts = line.split()
                if len(parts) >= 2:
                    current_name = parts[1]
            
            elif line.startswith('TYPE'):
                # 提取类型，例如 "TYPE        main cofac"
                current_type = line
            
            elif line == '///':
                # 检查是否为 main 类型
                if current_name and current_type and 'main' in current_type:
                    main_pairs.add(current_name)
                
                # 重置
                current_name = None
                current_type = None
    
    print(f"找到 {len(main_pairs)} 个 main 类型的代谢物对")
    
    # 1.2 解析 rclass 文件，构建邻接字典（保留 RC ID）
    print(f"正在解析 rclass 文件: {rclass_path}")
    adjacency_map = defaultdict(list)
    
    with open(rclass_path, 'r', encoding='utf-8') as f:
        current_rc_id = None
        current_rpairs = []
        
        for line in f:
            line = line.strip()
            
            if line.startswith('ENTRY'):
                # 提取 RC ID，例如 "ENTRY       RC00001"
                parts = line.split()
                if len(parts) >= 2:
                    current_rc_id = parts[1]
            
            elif line.startswith('RPAIR'):
                # 提取 RPAIR 行的所有代谢物对
                # 例如: "RPAIR       C00003_C00004    C00005_C00006"
                parts = line.split()[1:]  # 跳过 "RPAIR" 标签
                current_rpairs.extend(parts)
            
            elif not line.startswith('ENTRY') and not line.startswith('DEFINITION') and \
                 not line.startswith('REACTION') and not line.startswith('ENZYME') and \
                 not line.startswith('PATHWAY') and not line.startswith('ORTHOLOGY') and \
                 not line.startswith('///') and current_rpairs and line:
                # 继续读取多行 RPAIR 数据
                parts = line.split()
                for part in parts:
                    if '_' in part and part.startswith('C'):
                        current_rpairs.append(part)
            
            elif line == '///':
                # 处理当前 ENTRY 块
                if current_rc_id:
                    for rpair in current_rpairs:
                        # 检查是否在 main_pairs 中
                        if rpair in main_pairs:
                            # 提取代谢物对
                            if '_' in rpair:
                                metabolites = rpair.split('_')
                                if len(metabolites) == 2:
                                    met_a, met_b = metabolites
                                    # 创建连接对象
                                    conn_a_to_b = {"target": met_b, "rc_id": current_rc_id}
                                    conn_b_to_a = {"target": met_a, "rc_id": current_rc_id}
                                    
                                    # 双向添加（反应是可逆的），避免重复
                                    if conn_a_to_b not in adjacency_map[met_a]:
                                        adjacency_map[met_a].append(conn_a_to_b)
                                    if conn_b_to_a not in adjacency_map[met_b]:
                                        adjacency_map[met_b].append(conn_b_to_a)
                
                # 重置当前块
                current_rc_id = None
                current_rpairs = []
    
    # 转换为普通字典
    adjacency_map = dict(adjacency_map)
    
    print(f"构建完成！共有 {len(adjacency_map)} 个 KEGG 代谢物节点")
    total_connections = sum(len(connections) for connections in adjacency_map.values())
    print(f"共有 {total_connections // 2} 条边（无向图）")
    print(f"所有连接对象都包含 RC ID 信息")
    
    return adjacency_map


def parse_fgc_descriptions(fgc_path: str) -> Dict[str, str]:
    """
    Step 2: 解析反应功能描述
    
    Args:
        fgc_path: fgc02-reaction.keg 文件路径
    
    Returns:
        RC 编号到功能描述的映射 {RC_ID: description}
    """
    print("\n" + "=" * 60)
    print("Step 2: 解析反应功能描述")
    print("=" * 60)
    
    print(f"正在解析 fgc02-reaction.keg 文件: {fgc_path}")
    rc_map = {}
    current_description = None
    
    # 正则表达式：提取 RC 编号（格式：RC + 5位数字）
    rc_pattern = re.compile(r'RC\d{5}')
    
    with open(fgc_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 跳过元数据行
            if line.startswith('#') or line.startswith('+') or line.startswith('!'):
                continue
            
            # D 级：功能描述
            if line.startswith('D'):
                # 提取冒号后的描述
                # 例如: "D      C01AA: Aldehyde to Secondary alcohol"
                if ':' in line:
                    description = line.split(':', 1)[1].strip()
                    current_description = description
            
            # E 级：RC 编号
            elif line.startswith('E') and current_description:
                # 使用正则表达式提取 RC 编号
                # 例如: 'E        <a href="...">RC00371</a> ...'
                rc_matches = rc_pattern.findall(line)
                for rc_id in rc_matches:
                    rc_map[rc_id] = current_description
    
    print(f"解析完成！共找到 {len(rc_map)} 个 RC 功能描述")
    
    # 打印前 5 个示例
    print("\n示例（前5个）：")
    for i, (rc_id, desc) in enumerate(list(rc_map.items())[:5]):
        print(f"  {rc_id}: {desc}")
    
    return rc_map


def map_to_hmdb(kegg_graph: Dict[str, List[Dict[str, str]]], 
                mapping_csv_path: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Step 3: ID 转换与图谱迁移（增强版，保留 RC ID）
    
    Args:
        kegg_graph: KEGG 邻接字典 {KEGG_ID: [{"target": KEGG_ID, "rc_id": RC_ID}, ...]}
        mapping_csv_path: ID 映射表路径
    
    Returns:
        HMDB 邻接字典 {HMDB_ID: [{"target": HMDB_ID, "rc_id": RC_ID}, ...]}
    """
    print("\n" + "=" * 60)
    print("Step 3: ID 转换与图谱迁移（增强版）")
    print("=" * 60)
    
    # 3.1 加载 KEGG -> HMDB 映射表
    print(f"正在加载 ID 映射表: {mapping_csv_path}")
    kegg_to_hmdb = {}
    
    with open(mapping_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            kegg_id = row.get('KEGG', '').strip()
            hmdb_id = row.get('HMDB', '').strip()
            
            # 过滤无效值
            if kegg_id and hmdb_id and hmdb_id.upper() != 'NA':
                kegg_to_hmdb[kegg_id] = hmdb_id
    
    print(f"加载完成！共有 {len(kegg_to_hmdb)} 个 KEGG -> HMDB 映射")
    
    # 3.2 转换图谱（保留连接对象结构）
    print("正在转换图谱...")
    hmdb_graph = defaultdict(list)
    skipped_count = 0
    converted_connections = 0
    
    for kegg_a, connections in kegg_graph.items():
        # 查找 KEGG_A 对应的 HMDB ID
        hmdb_a = kegg_to_hmdb.get(kegg_a)
        
        if not hmdb_a:
            skipped_count += len(connections)
            continue
        
        for conn in connections:
            target_kegg = conn["target"]
            rc_id = conn["rc_id"]
            
            # 查找 target KEGG ID 对应的 HMDB ID
            hmdb_b = kegg_to_hmdb.get(target_kegg)
            
            if not hmdb_b:
                skipped_count += 1
                continue
            
            # 创建新的连接对象
            hmdb_conn = {"target": hmdb_b, "rc_id": rc_id}
            
            # 添加到 HMDB 图谱（避免重复）
            if hmdb_conn not in hmdb_graph[hmdb_a]:
                hmdb_graph[hmdb_a].append(hmdb_conn)
                converted_connections += 1
    
    # 转换为普通字典
    hmdb_graph = dict(hmdb_graph)
    
    print(f"转换完成！")
    print(f"  - HMDB 节点数: {len(hmdb_graph)}")
    print(f"  - 转换的连接数: {converted_connections}")
    print(f"  - 跳过的连接数（缺失 HMDB ID）: {skipped_count}")
    print(f"  - 所有连接对象都保留了 RC ID 信息")
    
    return hmdb_graph


def main():
    """
    主函数：执行完整的反应网络构建流程
    """
    print("\n" + "=" * 60)
    print("KEGG 反应网络构建脚本")
    print("=" * 60 + "\n")
    
    # 定义文件路径
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data"
    storage_dir = base_dir / "storage"
    
    rpair_path = data_dir / "rpair"
    rclass_path = data_dir / "rclass"
    fgc_path = data_dir / "fgc02-reaction.keg"
    mapping_csv_path = data_dir / "metabolitIDmapping.csv"
    
    # 确保存储目录存在
    storage_dir.mkdir(exist_ok=True)
    
    # Step 1: 构建原始 KEGG 反应图谱
    kegg_graph = build_raw_kegg_graph(str(rpair_path), str(rclass_path))
    
    # 保存原始 KEGG 图谱
    output_path_1 = storage_dir / "kegg_reaction_graph_raw.json"
    with open(output_path_1, 'w', encoding='utf-8') as f:
        json.dump(kegg_graph, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 已保存原始 KEGG 图谱到: {output_path_1}")
    
    # Step 2: 解析反应功能描述
    rc_map = parse_fgc_descriptions(str(fgc_path))
    
    # 保存 RC 功能描述
    output_path_2 = storage_dir / "reaction_class_info.json"
    with open(output_path_2, 'w', encoding='utf-8') as f:
        json.dump(rc_map, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 已保存 RC 功能描述到: {output_path_2}")
    
    # Step 3: ID 转换与图谱迁移
    hmdb_graph = map_to_hmdb(kegg_graph, str(mapping_csv_path))
    
    # 保存 HMDB 图谱
    output_path_3 = storage_dir / "hmdb_reaction_graph.json"
    with open(output_path_3, 'w', encoding='utf-8') as f:
        json.dump(hmdb_graph, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 已保存 HMDB 图谱到: {output_path_3}")
    
    # 最终统计
    print("\n" + "=" * 60)
    print("构建完成！最终统计：")
    print("=" * 60)
    print(f"原始 KEGG 节点数: {len(kegg_graph)}")
    print(f"转换后 HMDB 节点数: {len(hmdb_graph)}")
    print(f"未匹配节点数: {len(kegg_graph) - len(hmdb_graph)}")
    print(f"RC 功能描述数: {len(rc_map)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
