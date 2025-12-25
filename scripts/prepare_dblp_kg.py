import xml.sax
import os
import json
import re
from collections import defaultdict

DBLP_XML_PATH = 'data/real/dblp.xml'
OUTPUT_PATH = 'data/processed/kg_triples_ids.txt'
MAPPING_PATH = 'data/processed/kg_mappings.json'

RELATIONS = {
    "type": "rdf:type",
    "partOf": "dblp:partOf",             
    "listedIn": "dblp:listedIn",         
    
    "conferenceSeries": "dblp:conferenceSeries", 
    "conferenceYear": "dblp:conferenceYear",    
    "journalID": "dblp:journalID",               
    
    "author": "dblp:hasAuthor",
    "editor": "dblp:hasEditor",
    "primaryName": "dblp:primaryName",   
    "variantName": "dblp:variantName",   
    "homonymID": "dblp:homonymID",       
    "baseName": "dblp:baseName",         
    "coauthorWith": "dblp:coauthorWith", 
    "affiliation": "dblp:affiliation",
    
    "title": "dblp:title",
    "year": "dblp:publishedInYear",
    "month": "dblp:publishedInMonth",    
    "journal": "dblp:publishedInJournal",
    "booktitle": "dblp:presentedAt",
    "volume": "dblp:volume",
    "volumeFromKey": "dblp:volumeFromKey",
    "number": "dblp:issueNumber",
    "pages": "dblp:pages",
    "publisher": "dblp:publisher",
    "address": "dblp:address",           
    "isbn": "dblp:isbn",
    "issn": "dblp:issn",                 
    "series": "dblp:series",
    "seriesPage": "dblp:seriesPage",     
    "school": "dblp:school",
    "cdrom": "dblp:cdrom",               
    

    "ee": "dblp:hasDOI",
    "extractedDOI": "dblp:extractedDOI", 
    "url": "dblp:hasLink",
    "mdate": "dblp:lastModified",
    

    "cite": "dblp:cites"
}

TARGET_TAGS = {
    "article", "inproceedings", "proceedings", "book", "incollection", 
    "phdthesis", "mastersthesis", "www"
}

def parse_author_name(name):
    match = re.match(r'^(.+?)\s+(\d{4})$', name)
    if match:
        return match.groups()
    return name, None

def parse_dblp_key(key):
    parts = key.split('/')
    result = {'type': parts[0]}
    
    if parts[0] == 'conf' and len(parts) >= 3:
        result['series'] = parts[1]
        if parts[2].isdigit():
            result['year'] = parts[2]
        else:
            result['instance'] = parts[2]
            
    elif parts[0] == 'journals' and len(parts) >= 3:
        result['journal'] = parts[1]
        potential_vol = parts[2].replace(parts[1], '')
        if potential_vol.isdigit():
             result['volume'] = potential_vol
        else:
             result['instance'] = parts[2]
             
    elif parts[0] == 'homepages':
        result['person_path'] = '/'.join(parts[1:])
        
    return result

def extract_doi(ee_url):
    match = re.search(r'(10\.\d+/[^\s]+)', ee_url)
    return match.group(1) if match else None

class DBLPHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.count = 0
        self.current_tag = ""
        self.current_key = ""
        self.current_mdate = ""
        self.buffer = ""
        self.triples = []
        self.www_authors = [] 
        self.series_href = None 
        
    def startElement(self, tag, attributes):
        if tag in TARGET_TAGS:
            self.current_tag = tag
            self.current_key = attributes.get("key")
            self.current_mdate = attributes.get("mdate")
            
            if self.current_key:
                self.triples.append((self.current_key, RELATIONS["type"], "dblp:" + tag))
                if self.current_mdate:
                    self.triples.append((self.current_key, RELATIONS["mdate"], self.current_mdate))
                
                key_info = parse_dblp_key(self.current_key)
                
                if 'series' in key_info:
                    venue_entity = f"venue/conf/{key_info['series']}"
                    self.triples.append((self.current_key, RELATIONS["conferenceSeries"], venue_entity))
                    self.triples.append((venue_entity, RELATIONS["type"], "dblp:ConferenceSeries"))
                    self.triples.append((self.current_key, RELATIONS["listedIn"], venue_entity))
                    
                    if 'year' in key_info:
                         self.triples.append((self.current_key, RELATIONS["conferenceYear"], key_info['year']))

                if 'journal' in key_info:
                    journal_entity = f"venue/journal/{key_info['journal']}"
                    self.triples.append((self.current_key, RELATIONS["journalID"], journal_entity))
                    self.triples.append((journal_entity, RELATIONS["type"], "dblp:JournalSeries"))
                    self.triples.append((self.current_key, RELATIONS["listedIn"], journal_entity))
                    
                    if 'volume' in key_info:
                        self.triples.append((self.current_key, RELATIONS["volumeFromKey"], key_info['volume']))
                
                self.count += 1
        
        elif self.current_key:
            self.buffer = ""
            if tag == "series":
                self.series_href = attributes.get('href')

    def characters(self, content):
        if self.current_key:
            self.buffer += content

    def endElement(self, tag):
        if self.current_tag == "www":
            value = self.buffer.strip()
            if tag == "author":
                self.www_authors.append(value)
                return
            elif tag == "note" and value:
                self.triples.append((self.current_key, RELATIONS["affiliation"], value))
                return
            elif tag == "cite" and value:
                self.triples.append((self.current_key, RELATIONS["cite"], value))
                return

        if not self.current_key:
            return

        value = self.buffer.strip()
        
        if tag in TARGET_TAGS:
            if self.current_tag == "www" and self.current_key.startswith("homepages/"):
                if self.www_authors:
                    primary = self.www_authors[0]
                    self.triples.append((self.current_key, RELATIONS["primaryName"], primary))
                    for variant in self.www_authors[1:]:
                        self.triples.append((self.current_key, RELATIONS["variantName"], variant))
            self.www_authors = [] 
            
            self.current_key = ""
            self.current_tag = ""
            if self.count % 100000 == 0:
                print(f"Processed {self.count} publication records...")
            return

        if not value:
            return

        
        if tag == "author" or tag == "editor":
            base_name, homonym_id = parse_author_name(value)
            rel = RELATIONS["author"] if tag == "author" else RELATIONS["editor"]
            
            self.triples.append((self.current_key, rel, value))
            if homonym_id:
                self.triples.append((value, RELATIONS["homonymID"], homonym_id))
                self.triples.append((value, RELATIONS["baseName"], base_name))
                
        elif tag == "series":
            self.triples.append((self.current_key, RELATIONS["series"], value))
            if self.series_href:
                self.triples.append((value, RELATIONS["seriesPage"], self.series_href))
                self.series_href = None
                
        elif tag == "ee":
            self.triples.append((self.current_key, RELATIONS["ee"], value))
            doi = extract_doi(value)
            if doi:
                self.triples.append((self.current_key, RELATIONS["extractedDOI"], doi))
        
        elif tag == "title": self.triples.append((self.current_key, RELATIONS["title"], value))
        elif tag == "crossref": self.triples.append((self.current_key, RELATIONS["partOf"], value))
        elif tag == "journal": self.triples.append((self.current_key, RELATIONS["journal"], value))
        elif tag == "booktitle": self.triples.append((self.current_key, RELATIONS["booktitle"], value))
        elif tag == "year": self.triples.append((self.current_key, RELATIONS["year"], value))
        elif tag == "month": self.triples.append((self.current_key, RELATIONS["month"], value)) # Fix #7
        elif tag == "volume": self.triples.append((self.current_key, RELATIONS["volume"], value))
        elif tag == "number": self.triples.append((self.current_key, RELATIONS["number"], value))
        elif tag == "pages": self.triples.append((self.current_key, RELATIONS["pages"], value))
        elif tag == "publisher": self.triples.append((self.current_key, RELATIONS["publisher"], value))
        elif tag == "address": self.triples.append((self.current_key, RELATIONS["address"], value)) # Fix #6
        elif tag == "isbn": self.triples.append((self.current_key, RELATIONS["isbn"], value))
        elif tag == "issn": self.triples.append((self.current_key, RELATIONS["issn"], value)) # Fix #5
        elif tag == "school": self.triples.append((self.current_key, RELATIONS["school"], value))
        elif tag == "cdrom": self.triples.append((self.current_key, RELATIONS["cdrom"], value)) # Fix #8
        elif tag == "url": self.triples.append((self.current_key, RELATIONS["url"], value))
        elif tag == "cite": self.triples.append((self.current_key, RELATIONS["cite"], value))

def build_coauthor_graph(triples):
    """ Post-processing with Symmetric Edges (Fix #9) """
    print("Building Coauthor Graph (Symmetric)...")
    paper_authors = defaultdict(list)
    coauthor_edges = []
    
    for h, r, t in triples:
        if r == RELATIONS["author"]:
            paper_authors[h].append(t)
            
    for paper, authors in paper_authors.items():
        if len(authors) > 50: continue 
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                coauthor_edges.append((authors[i], RELATIONS["coauthorWith"], authors[j]))
                coauthor_edges.append((authors[j], RELATIONS["coauthorWith"], authors[i]))
    
    print(f"Added {len(coauthor_edges)} symmetric coauthor edges.")
    return coauthor_edges

def save_triples_and_mappings(triples, output_txt, output_json, max_entities=2000000):
    print(f"\n[FILTERING] Starting entity filtering (limit: {max_entities:,})...")
    print(f"Total triples before coauthor: {len(triples):,}")
    
    coauthor_triples = build_coauthor_graph(triples)
    all_triples = triples + coauthor_triples
    
    print(f"Total triples after coauthor graph: {len(all_triples):,}")
    
    print(f"[FILTERING] Counting entity frequencies...")
    entity_counts = defaultdict(int)
    relation_set = set()
    for h, r, t in all_triples:
        entity_counts[h] += 1
        entity_counts[t] += 1
        relation_set.add(r)
    
    print(f"[FILTERING] Found {len(entity_counts):,} unique entities")
    
    sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
    top_entities = set([e for e, _ in sorted_entities[:max_entities]])
    
    print(f"[FILTERING] Selecting top {max_entities:,} entities by frequency...")
    
    filtered_triples = []
    entity_set = set()
    for h, r, t in all_triples:
        if h in top_entities and t in top_entities:
            filtered_triples.append((h, r, t))
            entity_set.add(h)
            entity_set.add(t)
    
    print(f"[FILTERING] Filtered to {len(filtered_triples):,} triples (from {len(all_triples):,})")
    
    entity2id = {e: i for i, e in enumerate(sorted(entity_set))}
    relation2id = {r: i for i, r in enumerate(sorted(relation_set))}
    
    print(f"[SUCCESS] Final counts - Entities: {len(entity2id):,} | Relations: {len(relation2id)}")
    
    if len(entity2id) > max_entities:
        print(f"[WARN] Entity count ({len(entity2id):,}) exceeds limit ({max_entities:,})!")
    
    with open(output_txt, 'w', encoding='utf-8') as f:
        for h, r, t in filtered_triples:
            f.write(f"{entity2id[h]}\t{relation2id[r]}\t{entity2id[t]}\n")
            
    mappings = {
        "ent2id": entity2id,
        "rel2id": relation2id,
        "id2ent": {v: k for k, v in entity2id.items()},
        "id2rel": {v: k for k, v in relation2id.items()}
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=2)
    
    print("Optimization Complete: SCHEMA Generated.")

def main():
    if not os.path.exists('data/real/dblp.xml'):
        print("Error: data/real/dblp.xml not found!")
        return

    print("Parsing DBLP...")
    handler = DBLPHandler()
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)
    
    with open(DBLP_XML_PATH, 'r', encoding='iso-8859-1') as f:
        parser.parse(f)
        
    save_triples_and_mappings(handler.triples, OUTPUT_PATH, MAPPING_PATH)

if __name__ == "__main__":
    main()