"""
HMDB Chroma Vector Database Builder
Constructs a local Chroma vector database from cleaned HMDB metabolite data.
"""

import json
import os
import time
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()

# Configuration for retry logic
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
REQUEST_TIMEOUT = 300  # 5 minutes - increased for large batches


def load_metabolites(json_path: str) -> List[Dict]:
    """Load cleaned metabolite data from JSON file."""
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} metabolites.")
    return data


def create_document(metabolite: Dict) -> Document:
    """
    Create a LangChain Document from a metabolite dictionary.
    
    Optimized for performance:
    - Excludes pathways (too heavy, available in separate pathway database)
    - Truncates description to 500 chars
    - Limits synonyms and diseases to reduce token count
    
    Args:
        metabolite: Dictionary containing metabolite information
        
    Returns:
        Document with formatted page_content and metadata
    """
    # Extract fields with defaults
    name = metabolite.get('name', 'Unknown')
    hmdb_id = metabolite.get('hmdb_id', 'Unknown')
    description = metabolite.get('description', '')
    synonyms = metabolite.get('synonyms', [])
    diseases = metabolite.get('diseases', [])
    
    # Taxonomy fields
    super_class = metabolite.get('super_class') or 'Unclassified'
    class_name = metabolite.get('class') or 'Unclassified'
    sub_class = metabolite.get('sub_class') or 'Unclassified'
    direct_parent = metabolite.get('direct_parent') or 'Unclassified'
    
    # Limit synonyms to first 10 to reduce size
    synonyms_list = synonyms[:10] if len(synonyms) > 10 else synonyms
    synonyms_str = ', '.join(synonyms_list) if synonyms_list else 'None'
    if len(synonyms) > 10:
        synonyms_str += f' (and {len(synonyms) - 10} more)'
    
    # Limit diseases to first 5 to reduce size
    diseases_list = diseases[:5] if len(diseases) > 5 else diseases
    diseases_str = ', '.join(diseases_list) if diseases_list else 'None'
    if len(diseases) > 5:
        diseases_str += f' (and {len(diseases) - 5} more)'
    
    # Truncate description to 500 characters to reduce token count
    # Full descriptions can be 10,000+ characters
    description_truncated = description[:500] + '...' if len(description) > 500 else description
    
    # Build page content WITHOUT pathways (pathways are in separate PathBank database)
    page_content = f"""Name: {name}
ID: {hmdb_id}
Class: {class_name}
Synonyms: {synonyms_str}
Diseases: {diseases_str}
Description: {description_truncated}"""
    
    # Build metadata for filtering (also WITHOUT pathways)
    metadata = {
        'hmdb_id': hmdb_id,
        'name': name,
        'super_class': super_class,
        'class': class_name,
        'sub_class': sub_class,
        'direct_parent': direct_parent
    }
    
    return Document(page_content=page_content, metadata=metadata)


def build_vector_database(
    json_path: str,
    persist_directory: str,
    batch_size: int = 5000
) -> None:
    """
    Build Chroma vector database from cleaned metabolite data.
    
    Args:
        json_path: Path to cleaned JSON data
        persist_directory: Directory to persist the vector database
        batch_size: Number of documents to process in each batch
    """
    # Load data
    metabolites = load_metabolites(json_path)
    
    # Initialize embedding model with API base support
    print("Initializing OpenAI embedding model (text-embedding-3-small)...")
    
    embeddings_kwargs = {
        "model": "text-embedding-3-small",
        "max_retries": MAX_RETRIES,
        "request_timeout": REQUEST_TIMEOUT  # Use configured timeout
    }
    
    # Support for custom API base (e.g., proxy APIs)
    openai_api_base = os.getenv("OPENAI_API_BASE")
    if openai_api_base:
        embeddings_kwargs["openai_api_base"] = openai_api_base
        print(f"Using custom API base: {openai_api_base}")
    
    embeddings = OpenAIEmbeddings(**embeddings_kwargs)
    
    # Create documents
    print("Creating documents...")
    documents = []
    for metabolite in tqdm(metabolites, desc="Processing metabolites"):
        doc = create_document(metabolite)
        documents.append(doc)
    
    print(f"\nTotal documents created: {len(documents)}")
    
    # Ensure persist directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    # Build vector database in batches with retry logic
    print(f"\nBuilding vector database at {persist_directory}...")
    print(f"Processing in batches of {batch_size}...")
    
    vectordb = None
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(0, len(documents), batch_size), desc="Embedding batches", total=total_batches):
        batch = documents[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        
        # Retry logic for each batch
        for attempt in range(MAX_RETRIES):
            try:
                if vectordb is None:
                    # Create new vector store with first batch
                    vectordb = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=persist_directory
                    )
                    print(f"\n✓ Batch {batch_num}/{total_batches} completed (initial batch)")
                else:
                    # Add subsequent batches
                    vectordb.add_documents(batch)
                    print(f"\n✓ Batch {batch_num}/{total_batches} completed")
                
                # Success, break retry loop
                break
                
            except Exception as e:
                error_msg = str(e)
                print(f"\n⚠️  Batch {batch_num} attempt {attempt + 1}/{MAX_RETRIES} failed: {error_msg[:100]}")
                
                if attempt < MAX_RETRIES - 1:
                    print(f"   Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"\n❌ Batch {batch_num} failed after {MAX_RETRIES} attempts")
                    print(f"   Error: {error_msg}")
                    print(f"\n建议:")
                    print(f"   1. 检查 API 配额和限流")
                    print(f"   2. 尝试减小 batch_size (当前: {batch_size})")
                    print(f"   3. 检查网络连接")
                    print(f"   4. 考虑使用官方 OpenAI API")
                    raise
    
    # Explicit persist (for safety, though newer versions auto-persist)
    print("\nPersisting vector database...")
    if hasattr(vectordb, 'persist'):
        vectordb.persist()
    
    print(f"\n✓ 成功构建向量库，共存储 {len(documents)} 个代谢物。")
    print(f"✓ Vector database saved to: {persist_directory}")


if __name__ == "__main__":
    import sys
    
    # Configuration
    json_path = "data/hmdb_metabolites_serum_cleaned.json"
    persist_directory = "storage/hmdb_chroma"
    
    # Optimized batch size: 5000 (documents are now much smaller without pathways)
    batch_size = 5000
    
    print(f"Configuration:")
    print(f"  Input JSON: {json_path}")
    print(f"  Output directory: {persist_directory}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max retries per batch: {MAX_RETRIES}")
    print(f"  Retry delay: {RETRY_DELAY}s")
    print(f"  Request timeout: {REQUEST_TIMEOUT}s ({REQUEST_TIMEOUT//60} minutes)")
    
    # Check if output directory already exists
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"\n⚠️  警告: 输出目录已存在且非空: {persist_directory}")
        response = input("是否继续？这将覆盖现有数据 (y/n): ").strip().lower()
        if response != 'y':
            print("已取消操作")
            sys.exit(0)
    
    print()
    
    # Build database
    try:
        build_vector_database(
            json_path=json_path,
            persist_directory=persist_directory,
            batch_size=batch_size
        )
        print("\n🎉 向量数据库构建完成!")
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 构建失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
