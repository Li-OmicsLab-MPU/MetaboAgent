"""
PubMed 和 bioRxiv 文献检索模块

该模块作为 MetaboAgent 的"眼睛"，负责从 PubMed 和 bioRxiv 获取实时文献摘要。
支持根据不同的检索意图（association、mechanism）自动调整查询策略。

2026 Phase 0 upgrade:
- 新增 PubMed hit count 能力
- 新增 tiab 字段化查询构建
- 新增带元数据的摘要检索接口
- 新增面向 LLM 批处理抽取的摘要格式化接口

Author: MetaboAgent Team
Date: 2025-01-02
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from xml.etree import ElementTree as ET

import requests
from Bio import Entrez

# ============================================================================
# 配置 Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================================
# PubMedRetriever 类
# ============================================================================
class PubMedRetriever:
    """
    PubMed 和 bioRxiv 文献检索器
    
    功能：
    1. 从 PubMed 检索文献（使用 Bio.Entrez）
    2. 从 bioRxiv 补充检索结果
    3. 根据检索意图自动调整查询策略
    4. 格式化文献证据供 LLM 使用
    
    Attributes:
        email (str): Entrez API 所需的邮箱地址
        max_retries (int): API 调用失败时的最大重试次数
        retry_delay (float): 重试之间的延迟时间（秒）
    """
    
    # PubMed API 限制：每秒最多 3 个请求（无 API key）
    PUBMED_REQUEST_DELAY = 0.34  # 秒
    
    # bioRxiv API 端点
    BIORXIV_API_BASE = "https://api.biorxiv.org/details/biorxiv"
    
    # 意图相关的查询后缀
    INTENT_SUFFIXES = {
        "association": " AND (biomarker OR association)",
        "mechanism": " AND (pathway OR mechanism)",
        "general": ""
    }
    
    def __init__(
        self,
        email: str = "your_email@example.com",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        初始化 PubMedRetriever
        
        Args:
            email: Entrez API 所需的邮箱地址（占位符）
            max_retries: API 调用失败时的最大重试次数
            retry_delay: 重试之间的延迟时间（秒）
        """
        self.email = email
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 设置 Entrez 邮箱
        Entrez.email = self.email
        
        logger.info(f"PubMedRetriever initialized with email: {self.email}")

    # ========================================================================
    # 公共方法：查询构建与计数
    # ========================================================================
    def build_fielded_term(self, term: str, field: str = "tiab") -> str:
        """
        Build a quoted fielded term for PubMed queries.

        Examples:
            "lung cancer"[tiab]
            "glucose"[mesh]
        """
        clean_term = (term or "").strip()
        if not clean_term:
            return ""
        if not field:
            return f'"{clean_term}"'
        return f'"{clean_term}"[{field}]'

    def build_or_query(self, terms: Sequence[str], field: str = "tiab") -> str:
        """Join multiple terms with OR using a consistent PubMed field."""
        fielded_terms = [
            self.build_fielded_term(term, field=field)
            for term in terms
            if (term or "").strip()
        ]
        if not fielded_terms:
            return ""
        if len(fielded_terms) == 1:
            return fielded_terms[0]
        return "(" + " OR ".join(fielded_terms) + ")"

    def build_metabolite_disease_query(
        self,
        metabolite_terms: Sequence[str],
        disease_terms: Sequence[str],
        field: str = "tiab",
    ) -> str:
        """
        Build a canonical metabolite-disease query.

        Example:
            ("glucose"[tiab] OR "d-glucose"[tiab]) AND ("lung cancer"[tiab])
        """
        metabolite_query = self.build_or_query(metabolite_terms, field=field)
        disease_query = self.build_or_query(disease_terms, field=field)

        if metabolite_query and disease_query:
            return f"{metabolite_query} AND {disease_query}"
        return metabolite_query or disease_query

    def count_pubmed_hits(self, query: str, intent: str = "general") -> int:
        """
        Count PubMed hits without downloading abstracts.

        Args:
            query: Base query string
            intent: Query intent suffix to apply

        Returns:
            Integer hit count. Returns 0 on failure.
        """
        try:
            adjusted_query = self._adjust_query_by_intent(query, intent)
            logger.info(f"Counting PubMed hits for query: '{adjusted_query}'")
            search_handle = Entrez.esearch(
                db="pubmed",
                term=adjusted_query,
                retmax=0,
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            count_str = search_results.get("Count", "0")
            return int(count_str)
        except Exception as e:
            logger.error(f"Error counting PubMed hits: {e}", exc_info=True)
            return 0

    def count_hits(self, query: str, source: str = "pubmed", intent: str = "general") -> int:
        """
        Generic hit-count wrapper.

        Currently PubMed is authoritative. bioRxiv returns a best-effort count
        based on the filtered API window.
        """
        if source.lower() == "pubmed":
            return self.count_pubmed_hits(query, intent=intent)
        if source.lower() == "biorxiv":
            return self._count_biorxiv_hits(query)
        raise ValueError(f"Unsupported source for hit count: {source}")
    
    def search(
        self,
        query: str,
        max_results: int = 5,
        intent: str = "general"
    ) -> List[Dict[str, Any]]:
        """
        检索文献（主方法）
        
        根据检索意图自动调整查询，优先从 PubMed 检索，不足时从 bioRxiv 补充。
        
        Args:
            query: 检索查询字符串
            max_results: 最大返回结果数
            intent: 检索意图，可选值：
                - "association": 关联研究（自动添加 biomarker/association 关键词）
                - "mechanism": 机制研究（自动添加 pathway/mechanism 关键词）
                - "general": 通用检索（不添加额外关键词）
        
        Returns:
            文献列表，每篇文献包含以下字段：
            - title (str): 标题
            - abstract (str): 摘要
            - year (int): 发表年份
            - source (str): 来源（"PubMed" 或 "bioRxiv"）
            - id (str): PMID 或 DOI
        
        Example:
            >>> retriever = PubMedRetriever()
            >>> papers = retriever.search("metabolomics diabetes", max_results=5, intent="association")
            >>> print(f"Found {len(papers)} papers")
        """
        result = self.search_with_metadata(
            query=query,
            max_results=max_results,
            intent=intent,
            sort="relevance",
            years_back=None,
            include_biorxiv=True,
        )
        return result.get("papers", [])

    def search_with_metadata(
        self,
        query: str,
        max_results: int = 5,
        intent: str = "general",
        sort: str = "relevance",
        years_back: Optional[int] = None,
        include_biorxiv: bool = True,
    ) -> Dict[str, Any]:
        """
        Search literature and return both papers and retrieval metadata.

        Args:
            query: Base query string
            max_results: Maximum number of papers to retrieve
            intent: Retrieval intent
            sort: PubMed sort mode, usually "relevance" or "pub date"
            years_back: Optional trailing-year filter
            include_biorxiv: Whether to backfill with bioRxiv when PubMed is short

        Returns:
            Dict with query metadata and `papers`.
        """
        logger.info(
            f"Starting metadata search with query='{query}', max_results={max_results}, "
            f"intent='{intent}', sort='{sort}', years_back={years_back}"
        )

        adjusted_query = self._adjust_query_by_intent(query, intent)
        pubmed_hit_count = self.count_pubmed_hits(query, intent=intent)
        pubmed_papers = self._search_pubmed(
            adjusted_query,
            max_results=max_results,
            sort=sort,
            years_back=years_back,
        )

        biorxiv_papers: List[Dict[str, Any]] = []
        if include_biorxiv and len(pubmed_papers) < max_results:
            remaining = max_results - len(pubmed_papers)
            logger.info(f"PubMed results insufficient, fetching {remaining} more from bioRxiv")
            biorxiv_papers = self._search_biorxiv(query, remaining)

        papers = pubmed_papers + biorxiv_papers
        return {
            "query": query,
            "adjusted_query": adjusted_query,
            "intent": intent,
            "sort": sort,
            "years_back": years_back,
            "max_results": max_results,
            "pubmed_hit_count": pubmed_hit_count,
            "pubmed_retrieved_count": len(pubmed_papers),
            "biorxiv_retrieved_count": len(biorxiv_papers),
            "retrieved_count": len(papers),
            "papers": papers,
        }

    def search_tiab_metabolite_disease(
        self,
        metabolite_terms: Sequence[str],
        disease_terms: Sequence[str],
        max_results: int = 10,
        intent: str = "general",
        years_back: Optional[int] = None,
        sort: str = "relevance",
    ) -> Dict[str, Any]:
        """
        Search PubMed using a canonical `tiab` metabolite-disease query.

        This is the main retrieval entry point for the upgraded Phase 0.
        """
        query = self.build_metabolite_disease_query(
            metabolite_terms=metabolite_terms,
            disease_terms=disease_terms,
            field="tiab",
        )
        return self.search_with_metadata(
            query=query,
            max_results=max_results,
            intent=intent,
            sort=sort,
            years_back=years_back,
            include_biorxiv=False,
        )
    
    def format_evidence(self, papers: List[Dict[str, Any]]) -> str:
        """
        格式化文献证据为 LLM 可读的字符串
        
        将文献列表格式化为结构化字符串，按年份降序排列（最新的在前）。
        
        Args:
            papers: 文献列表（由 search 方法返回）
        
        Returns:
            格式化的证据字符串，格式如下：
            [Evidence 1 | PubMed | 2024]
            Title: ...
            Abstract: ...
            
            [Evidence 2 | bioRxiv | 2023]
            Title: ...
            Abstract: ...
        
        Example:
            >>> papers = retriever.search("metabolomics", max_results=3)
            >>> evidence_text = retriever.format_evidence(papers)
            >>> print(evidence_text)
        """
        if not papers:
            logger.warning("No papers to format")
            return "No evidence available."
        
        # 步骤1: 按年份降序排序（最新的在前）
        sorted_papers = sorted(papers, key=lambda p: p.get("year", 0), reverse=True)
        logger.info(f"Formatting {len(sorted_papers)} papers as evidence")
        
        # 步骤2: 格式化每篇文献
        evidence_lines: List[str] = []
        for idx, paper in enumerate(sorted_papers, start=1):
            title = paper.get("title", "No title")
            abstract = paper.get("abstract", "No abstract available")
            year = paper.get("year", "Unknown")
            source = paper.get("source", "Unknown")
            paper_id = paper.get("id", "Unknown")
            
            # 格式化单篇文献
            evidence_block = (
                f"[Evidence {idx} | {source} | {year}]\n"
                f"ID: {paper_id}\n"
                f"Title: {title}\n"
                f"Abstract: {abstract}\n"
            )
            evidence_lines.append(evidence_block)
        
        # 步骤3: 合并所有文献
        formatted_evidence = "\n".join(evidence_lines)
        logger.info("Evidence formatting completed")
        
        return formatted_evidence

    def format_batch_abstracts_for_llm(
        self,
        papers: List[Dict[str, Any]],
        max_abstract_chars: Optional[int] = None,
    ) -> str:
        """
        Format papers as a compact list for batch abstract extraction prompts.

        The output is intentionally deterministic and easy for another LLM to
        parse into study-level structured evidence.
        """
        if not papers:
            return "[]"

        sorted_papers = sorted(papers, key=lambda p: p.get("year", 0), reverse=True)
        lines: List[str] = ["["]
        for idx, paper in enumerate(sorted_papers, start=1):
            abstract = paper.get("abstract", "") or ""
            if max_abstract_chars is not None and max_abstract_chars > 0:
                abstract = abstract[:max_abstract_chars]

            lines.extend([
                "  {",
                f'    "index": {idx},',
                f'    "id": "{paper.get("id", "Unknown")}",',
                f'    "source": "{paper.get("source", "Unknown")}",',
                f'    "year": {int(paper.get("year", 0) or 0)},',
                f'    "title": {self._json_quote(paper.get("title", "No title"))},',
                f'    "abstract": {self._json_quote(abstract or "No abstract available")}',
                "  },",
            ])

        if lines[-1] == "  },":
            lines[-1] = "  }"
        elif lines[-1].endswith(","):
            lines[-1] = lines[-1][:-1]
        lines.append("]")
        return "\n".join(lines)
    
    # ========================================================================
    # 私有方法：查询调整
    # ========================================================================
    def _adjust_query_by_intent(self, query: str, intent: str) -> str:
        """
        根据检索意图调整查询字符串
        
        Args:
            query: 原始查询
            intent: 检索意图
        
        Returns:
            调整后的查询字符串
        """
        suffix = self.INTENT_SUFFIXES.get(intent, "")
        if suffix:
            return query + suffix
        return query

    def _json_quote(self, value: Any) -> str:
        """Safely JSON-quote a scalar for prompt formatting."""
        import json
        return json.dumps("" if value is None else str(value), ensure_ascii=False)
    
    # ========================================================================
    # 私有方法：PubMed 检索
    # ========================================================================
    def _search_pubmed(
        self,
        query: str,
        max_results: int,
        sort: str = "relevance",
        years_back: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        从 PubMed 检索文献
        
        使用 Bio.Entrez 的 esearch 和 efetch 方法。
        
        Args:
            query: 检索查询
            max_results: 最大结果数
        
        Returns:
            文献列表
        """
        papers: List[Dict[str, Any]] = []
        
        try:
            # 步骤1: 使用 esearch 获取 PMID 列表
            logger.info(f"Calling PubMed esearch with query: '{query}'")
            esearch_kwargs: Dict[str, Any] = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "sort": sort,
            }
            if years_back is not None and years_back > 0:
                esearch_kwargs["datetype"] = "pdat"
                esearch_kwargs["mindate"] = str(datetime.now().year - years_back)
                esearch_kwargs["maxdate"] = str(datetime.now().year)

            search_handle = Entrez.esearch(**esearch_kwargs)
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            id_list = search_results.get("IdList", [])
            logger.info(f"PubMed esearch returned {len(id_list)} IDs")
            
            if not id_list:
                logger.warning("No results found in PubMed")
                return papers
            
            # 步骤2: 使用 efetch 获取文献详情
            time.sleep(self.PUBMED_REQUEST_DELAY)  # 遵守 API 限制
            
            logger.info(f"Calling PubMed efetch for {len(id_list)} IDs")
            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=id_list,
                rettype="xml",
                retmode="xml"
            )
            fetch_results = fetch_handle.read()
            fetch_handle.close()
            
            # 步骤3: 解析 XML 结果
            papers = self._parse_pubmed_xml(fetch_results)
            logger.info(f"Successfully parsed {len(papers)} papers from PubMed")
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}", exc_info=True)
            # 不抛出异常，返回空列表
        
        return papers
    
    def _parse_pubmed_xml(self, xml_data: str) -> List[Dict[str, Any]]:
        """
        解析 PubMed XML 响应
        
        Args:
            xml_data: XML 字符串
        
        Returns:
            文献列表
        """
        papers: List[Dict[str, Any]] = []
        
        try:
            root = ET.fromstring(xml_data)
            
            for article in root.findall(".//PubmedArticle"):
                try:
                    # 提取 PMID
                    pmid_elem = article.find(".//PMID")
                    pmid = pmid_elem.text if pmid_elem is not None else "Unknown"
                    
                    # 提取标题
                    title_elem = article.find(".//ArticleTitle")
                    title = self._extract_xml_text(title_elem) if title_elem is not None else "No title"
                    
                    # 提取摘要（兼容多个 AbstractText 分段）
                    abstract_nodes = article.findall(".//Abstract/AbstractText")
                    if abstract_nodes:
                        abstract_parts = []
                        for node in abstract_nodes:
                            label = node.attrib.get("Label")
                            text = self._extract_xml_text(node)
                            if not text:
                                continue
                            if label:
                                abstract_parts.append(f"{label}: {text}")
                            else:
                                abstract_parts.append(text)
                        abstract = " ".join(abstract_parts) if abstract_parts else "No abstract available"
                    else:
                        abstract = "No abstract available"

                    publication_types = [
                        self._extract_xml_text(node)
                        for node in article.findall(".//PublicationType")
                        if self._extract_xml_text(node)
                    ]
                    journal_elem = article.find(".//Journal/Title")
                    journal = self._extract_xml_text(journal_elem) if journal_elem is not None else ""
                    
                    # 提取年份
                    year_elem = article.find(".//PubDate/Year")
                    year = int(year_elem.text) if year_elem is not None else 0
                    
                    # 如果年份为 0，尝试从 MedlineDate 提取
                    if year == 0:
                        medline_date_elem = article.find(".//PubDate/MedlineDate")
                        if medline_date_elem is not None:
                            medline_date = medline_date_elem.text
                            # 尝试提取前 4 位数字作为年份
                            year_str = "".join(filter(str.isdigit, medline_date[:4]))
                            year = int(year_str) if year_str else 0
                    
                    paper = {
                        "title": title,
                        "abstract": abstract,
                        "year": year,
                        "source": "PubMed",
                        "id": pmid,
                        "journal": journal,
                        "publication_types": publication_types,
                    }
                    papers.append(paper)
                    
                except Exception as e:
                    logger.warning(f"Error parsing individual PubMed article: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing PubMed XML: {e}", exc_info=True)
        
        return papers

    def _extract_xml_text(self, node: Optional[ET.Element]) -> str:
        """Extract full text from an XML node including nested tags."""
        if node is None:
            return ""
        return "".join(node.itertext()).strip()
    
    # ========================================================================
    # 私有方法：bioRxiv 检索
    # ========================================================================
    def _search_biorxiv(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        从 bioRxiv 检索文献
        
        使用 bioRxiv 官方 API。
        注意：bioRxiv API 的搜索功能有限，这里使用日期范围检索。
        
        Args:
            query: 检索查询（注意：bioRxiv API 不支持全文搜索）
            max_results: 最大结果数
        
        Returns:
            文献列表
        """
        papers: List[Dict[str, Any]] = []
        
        try:
            # bioRxiv API 限制：需要指定日期范围
            # 这里使用最近 30 天的文献作为示例
            # 实际应用中可能需要更复杂的策略
            
            logger.info(f"Calling bioRxiv API (note: limited search capabilities)")
            
            # 构建 API URL（获取最近的文献）
            # 格式：https://api.biorxiv.org/details/biorxiv/2024-01-01/2024-01-31
            # 由于 API 限制，这里简化为获取最近的文献
            
            # 注意：bioRxiv API 实际上不支持关键词搜索
            # 这里提供一个基础实现，实际使用时可能需要调整
            url = f"{self.BIORXIV_API_BASE}/2023-01-01/2024-12-31/0/100"
            
            response = self._make_request_with_retry(url)
            
            if response and response.status_code == 200:
                data = response.json()
                collection = data.get("collection", [])
                
                # 简单过滤：标题或摘要包含查询关键词
                query_lower = query.lower()
                filtered_papers = []
                
                for item in collection:
                    title = item.get("title", "").lower()
                    abstract = item.get("abstract", "").lower()
                    
                    # 简单的关键词匹配
                    if query_lower in title or query_lower in abstract:
                        filtered_papers.append(item)
                    
                    if len(filtered_papers) >= max_results:
                        break
                
                # 解析结果
                papers = self._parse_biorxiv_response(filtered_papers[:max_results])
                logger.info(f"Successfully retrieved {len(papers)} papers from bioRxiv")
            else:
                logger.warning(f"bioRxiv API returned status code: {response.status_code if response else 'None'}")
        
        except Exception as e:
            logger.error(f"Error searching bioRxiv: {e}", exc_info=True)
            # 不抛出异常，返回空列表
        
        return papers

    def _count_biorxiv_hits(self, query: str) -> int:
        """
        Best-effort bioRxiv hit count.

        The official bioRxiv API does not support exact keyword hit counting, so
        this method counts matches within the currently scanned API window.
        """
        try:
            papers = self._search_biorxiv(query=query, max_results=100)
            return len(papers)
        except Exception as e:
            logger.error(f"Error counting bioRxiv hits: {e}", exc_info=True)
            return 0
    
    def _parse_biorxiv_response(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        解析 bioRxiv API 响应
        
        Args:
            items: bioRxiv API 返回的文献列表
        
        Returns:
            标准化的文献列表
        """
        papers: List[Dict[str, Any]] = []
        
        for item in items:
            try:
                title = item.get("title", "No title")
                abstract = item.get("abstract", "No abstract available")
                doi = item.get("doi", "Unknown")
                
                # 提取年份
                date_str = item.get("date", "")
                year = 0
                if date_str:
                    try:
                        year = int(date_str.split("-")[0])
                    except (ValueError, IndexError):
                        year = 0
                
                paper = {
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "source": "bioRxiv",
                    "id": doi
                }
                papers.append(paper)
                
            except Exception as e:
                logger.warning(f"Error parsing individual bioRxiv item: {e}")
                continue
        
        return papers
    
    # ========================================================================
    # 私有方法：HTTP 请求工具
    # ========================================================================
    def _make_request_with_retry(
        self,
        url: str,
        method: str = "GET",
        **kwargs
    ) -> Optional[requests.Response]:
        """
        带重试机制的 HTTP 请求
        
        Args:
            url: 请求 URL
            method: HTTP 方法（GET/POST）
            **kwargs: 传递给 requests 的其他参数
        
        Returns:
            Response 对象，失败时返回 None
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Making {method} request to {url} (attempt {attempt}/{self.max_retries})")
                
                if method.upper() == "GET":
                    response = requests.get(url, timeout=10, **kwargs)
                elif method.upper() == "POST":
                    response = requests.post(url, timeout=10, **kwargs)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed for URL: {url}")
        
        return None


# ============================================================================
# 使用示例
# ============================================================================
if __name__ == "__main__":
    # 创建检索器实例
    retriever = PubMedRetriever(email="your_email@example.com")
    
    # 示例1: 通用检索
    print("\n" + "="*80)
    print("示例1: 通用检索")
    print("="*80)
    papers = retriever.search("metabolomics diabetes", max_results=3, intent="general")
    print(f"\n找到 {len(papers)} 篇文献")
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. [{paper['source']}] {paper['title'][:80]}...")
        print(f"   Year: {paper['year']}, ID: {paper['id']}")
    
    # 示例2: 关联研究检索
    print("\n" + "="*80)
    print("示例2: 关联研究检索（自动添加 biomarker/association 关键词）")
    print("="*80)
    papers = retriever.search("glucose metabolism", max_results=2, intent="association")
    evidence = retriever.format_evidence(papers)
    print("\n格式化的证据：")
    print(evidence)
    
    # 示例3: 机制研究检索
    print("\n" + "="*80)
    print("示例3: 机制研究检索（自动添加 pathway/mechanism 关键词）")
    print("="*80)
    papers = retriever.search("insulin resistance", max_results=2, intent="mechanism")
    evidence = retriever.format_evidence(papers)
    print("\n格式化的证据：")
    print(evidence)
