import pandas as pd
import torch
import json
import plotly.express as px
from pathlib import Path
import os
import sys
from sklearn.decomposition import PCA

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

CHECKPOINT_PATH = Path(parent_dir) / "checkpoints/gan_latest.pth"
MAPPINGS_FILE = Path(parent_dir) / "data/processed/kg_mappings.json"
OUTPUT_HTML = Path(parent_dir) / "semantic_map.html"

def create_interactive_map():
    if not CHECKPOINT_PATH.exists():
        print(f"Error: {CHECKPOINT_PATH} not found.")
        return

    print("Loading mappings...")
    with open(MAPPINGS_FILE, "r") as f:
        id_to_str = json.load(f)
    
    print("Loading model checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    embeddings = checkpoint["D_state"]["ent_embedding.weight"].numpy()

    indices = []
    metadata = []
    
    print("Selecting entities for visualization...")
    for eid, text in id_to_str.items():
        if "_" not in eid: continue
        try:
            idx = int(eid.split("_")[1])
        except ValueError: continue

        if idx < len(embeddings) and len(indices) < 5000:
            indices.append(idx)
            
            
            if eid.startswith("venue_"): e_type = "Conference"
            elif eid.startswith("author_"): e_type = "Author"
            elif eid.startswith("doi_"): e_type = "DOI"
            elif eid.startswith("link_"): e_type = "Link"
            elif eid.startswith("type_"): e_type = "Paper Type"
            elif eid.startswith("collection_"): e_type = "Collection"
            else: e_type = "Publication"
            
            metadata.append({"Name": text, "Type": e_type})

    selected_embeddings = embeddings[indices]
    
    print("Running PCA...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(selected_embeddings)

    df = pd.DataFrame(metadata)
    df["X"] = coords[:, 0]
    df["Y"] = coords[:, 1]

    print("Generating Plotly map...")
    fig = px.scatter(
        df, x="X", y="Y", color="Type", hover_name="Name",
        template="plotly_dark",
        color_discrete_map={
            "Conference": "#00ffff",  
            "Author": "#ff0000",      
            "DOI": "#00ff00",         
            "Link": "#ffff00",        
            "Paper Type": "#ff00ff",  
            "Collection": "#ffa500",  
            "Publication": "#808080"  
        }
    )

    fig.update_traces(marker=dict(size=5, opacity=0.7, line=dict(width=0.5, color='white')))
    fig.update_layout(showlegend=True, legend_title_text='Entity Category')
    
    fig.write_html(str(OUTPUT_HTML))
    print(f"Interactive map generated: {OUTPUT_HTML}")

if __name__ == "__main__":
    create_interactive_map()