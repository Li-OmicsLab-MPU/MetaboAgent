"""
HMDB XML to JSON Cleaner
Converts HMDB metabolites XML data to structured JSON format for vector database ingestion.

Field Extraction Strategy:
- Core fields (name, accession, description): Direct children only (safety first)
- Structure fields (taxonomy, diseases, synonyms): Recursive search for container tags
- Pathways: Recursive search (nested in biological_properties)

Supports both direct XML files and ZIP archives.
"""

import json
import zipfile
from pathlib import Path
from lxml import etree
from typing import List, Dict, Optional, Union


def clean_text(text: Optional[str]) -> Optional[str]:
    """Clean text by removing extra whitespace and newlines."""
    if text is None:
        return None
    return ' '.join(text.strip().split())


def extract_list_field(parent, tag: str, namespace: str) -> List[str]:
    """Extract a list of text values from child elements."""
    if parent is None:
        return []
    
    items = []
    for elem in parent.findall(f'.//{{{namespace}}}{tag}'):
        text = elem.text
        if text and text.strip():
            items.append(text.strip())
    return items


def parse_metabolite(metabolite_elem, namespace: str) -> Dict:
    """
    Parse a single metabolite element and extract required fields.
    
    Uses field-specific extraction strategies:
    - Core fields: Direct children only (prevents grabbing nested pathway names, etc.)
    - Container fields: Recursive search (unique container names)
    - Pathways: Recursive search (nested in biological_properties)
    """
    
    def get_text_direct(tag: str, parent=None) -> Optional[str]:
        """Get text from DIRECT child only (no recursion)."""
        elem = (parent if parent is not None else metabolite_elem).find(f'{{{namespace}}}{tag}')
        return elem.text if elem is not None and elem.text else None
    
    def get_text_recursive(tag: str, parent=None) -> Optional[str]:
        """Get text from anywhere in tree (recursive search)."""
        elem = (parent if parent is not None else metabolite_elem).find(f'.//{{{namespace}}}{tag}')
        return elem.text if elem is not None and elem.text else None
    
    # CORE FIELDS: Direct children only for safety
    hmdb_id = get_text_direct('accession')
    name = get_text_direct('name')
    description = clean_text(get_text_direct('description'))
    
    # SYNONYMS: Recursive search for container, then extract list
    synonyms_elem = metabolite_elem.find(f'.//{{{namespace}}}synonyms')
    synonyms = extract_list_field(synonyms_elem, 'synonym', namespace) if synonyms_elem is not None else []
    
    # TAXONOMY: Recursive search for container (unique name)
    taxonomy_elem = metabolite_elem.find(f'.//{{{namespace}}}taxonomy')
    if taxonomy_elem is not None:
        super_class = get_text_direct('super_class', taxonomy_elem)
        class_name = get_text_direct('class', taxonomy_elem)
        sub_class = get_text_direct('sub_class', taxonomy_elem)
        direct_parent = get_text_direct('direct_parent', taxonomy_elem)
    else:
        super_class = None
        class_name = None
        sub_class = None
        direct_parent = None
    
    # DISEASES: Recursive search for container
    diseases_elem = metabolite_elem.find(f'.//{{{namespace}}}diseases')
    diseases = []
    if diseases_elem is not None:
        for disease in diseases_elem.findall(f'{{{namespace}}}disease'):
            disease_name = get_text_direct('name', disease)
            if disease_name:
                diseases.append(disease_name)
    
    # PATHWAYS: Recursive search (nested in biological_properties)
    pathways_elem = metabolite_elem.find(f'.//{{{namespace}}}pathways')
    pathways = []
    if pathways_elem is not None:
        for pathway in pathways_elem.findall(f'{{{namespace}}}pathway'):
            pathway_name = get_text_direct('name', pathway)
            smpdb_id = get_text_direct('smpdb_id', pathway)
            kegg_map_id = get_text_direct('kegg_map_id', pathway)
            
            if pathway_name:
                pathway_data = {'name': pathway_name}
                if smpdb_id:
                    pathway_data['smpdb_id'] = smpdb_id
                if kegg_map_id:
                    pathway_data['kegg_map_id'] = kegg_map_id
                pathways.append(pathway_data)
    
    return {
        'hmdb_id': hmdb_id,
        'name': name,
        'description': description,
        'synonyms': synonyms,
        'super_class': super_class,
        'class': class_name,
        'sub_class': sub_class,
        'direct_parent': direct_parent,
        'diseases': diseases,
        'pathways': pathways
    }


def parse_hmdb_xml(xml_source: Union[str, Path], json_path: str) -> None:
    """
    Parse HMDB XML file (or ZIP archive) and convert to cleaned JSON format.
    
    Args:
        xml_source: Path to input XML file or ZIP archive containing XML
        json_path: Path to output JSON file
    """
    namespace = 'http://www.hmdb.ca'
    metabolites = []
    count = 0
    pathways_found = 0
    
    xml_source = Path(xml_source)
    print(f"Starting to parse {xml_source}...")
    print(f"Using namespace: {namespace}")
    
    # Determine if we're reading from ZIP or direct XML
    if xml_source.suffix == '.zip':
        print(f"Detected ZIP archive, extracting XML...")
        with zipfile.ZipFile(xml_source, 'r') as zip_ref:
            # Find the XML file in the ZIP (usually hmdb_metabolites.xml)
            xml_files = [f for f in zip_ref.namelist() if f.endswith('.xml')]
            if not xml_files:
                raise ValueError(f"No XML file found in {xml_source}")
            
            xml_filename = xml_files[0]
            print(f"Found XML file in archive: {xml_filename}")
            
            with zip_ref.open(xml_filename) as xml_file:
                context = etree.iterparse(xml_file, events=('end',), tag=f'{{{namespace}}}metabolite')
                metabolites, count, pathways_found = _process_metabolites(context, namespace)
    else:
        # Direct XML file
        context = etree.iterparse(str(xml_source), events=('end',), tag=f'{{{namespace}}}metabolite')
        metabolites, count, pathways_found = _process_metabolites(context, namespace)
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total metabolites processed: {count}")
    print(f"Metabolites with pathways: {pathways_found}")
    print(f"Pathway extraction rate: {pathways_found/count*100:.1f}%")
    print(f"{'='*60}\n")
    
    print(f"Writing to {json_path}...")
    
    # Write to JSON file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metabolites, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully wrote {len(metabolites)} metabolites to {json_path}")


def _process_metabolites(context, namespace: str) -> tuple:
    """
    Process metabolites from iterparse context.
    Returns: (metabolites_list, count, pathways_found_count)
    """
    metabolites = []
    count = 0
    pathways_found = 0
    
    for event, elem in context:
        try:
            metabolite_data = parse_metabolite(elem, namespace)
            metabolites.append(metabolite_data)
            count += 1
            
            # Track pathways extraction
            if metabolite_data.get('pathways'):
                pathways_found += 1
            
            # Verification output every 1000 records
            if count % 1000 == 0:
                print(f"Processed {count} metabolites...")
                # Show sample data for verification
                if metabolite_data.get('name'):
                    tax_class = metabolite_data.get('class', 'N/A')
                    pathway_count = len(metabolite_data.get('pathways', []))
                    print(f"  Sample: [{metabolite_data['name']}] | Taxonomy: {tax_class} | Pathways: {pathway_count}")
            
        except Exception as e:
            print(f"Error processing metabolite at count {count}: {e}")
            # Continue processing other metabolites
        
        # Clear element to free memory
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    
    del context
    return metabolites, count, pathways_found


if __name__ == "__main__":
    import sys
    
    # Support both serum and full HMDB datasets
    if len(sys.argv) > 1:
        xml_source = sys.argv[1]
        json_output = sys.argv[2] if len(sys.argv) > 2 else "data/hmdb_metabolites_cleaned.json"
    else:
        # Default: process serum metabolites
        xml_source = "data/serum_metabolites.xml"
        json_output = "data/hmdb_metabolites_cleaned.json"
        
        # Check if full HMDB ZIP exists
        full_hmdb_zip = Path("data/hmdb_metabolites.zip")
        if full_hmdb_zip.exists():
            print(f"\nDetected full HMDB archive: {full_hmdb_zip}")
            response = input("Process full HMDB dataset instead of serum only? (y/n): ").strip().lower()
            if response == 'y':
                xml_source = str(full_hmdb_zip)
                json_output = "data/hmdb_metabolites_full_cleaned.json"
    
    print(f"\nConfiguration:")
    print(f"  Input:  {xml_source}")
    print(f"  Output: {json_output}")
    print()
    
    parse_hmdb_xml(xml_source, json_output)
    
    # Verification: Load and check first few records
    print(f"\n{'='*60}")
    print("DATA INTEGRITY VERIFICATION")
    print(f"{'='*60}")
    
    with open(json_output, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nShowing first 5 metabolites with pathway data:\n")
    shown = 0
    for record in data:
        if record.get('pathways') and shown < 5:
            name = record.get('name', 'N/A')
            tax_class = record.get('class', 'N/A')
            pathway_count = len(record.get('pathways', []))
            pathway_names = [p.get('name', p) if isinstance(p, dict) else p 
                           for p in record.get('pathways', [])][:2]
            
            print(f"{shown + 1}. Name: {name}")
            print(f"   Taxonomy Class: {tax_class}")
            print(f"   Pathways ({pathway_count}): {', '.join(pathway_names)}")
            if pathway_count > 2:
                print(f"   ... and {pathway_count - 2} more")
            print()
            shown += 1
    
    if shown == 0:
        print("⚠️  WARNING: No metabolites with pathway data found!")
        print("   This may indicate an extraction issue.")
    else:
        print(f"✓ Verification complete: Pathway data successfully extracted")
    
    print(f"{'='*60}")
