import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import json

class DBLPDataset(Dataset):
    def __init__(self, triples_path, mappings_path=None):
        print(f"Loading dataset from {triples_path}...")
        
        try:
            df = pd.read_csv(
                triples_path, 
                sep='\t', 
                header=None, 
                names=['h', 'r', 't'], 
                dtype={'h': np.int32, 'r': np.int32, 't': np.int32},
                engine='c' 
            )
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise e

        self.data = torch.tensor(df.values, dtype=torch.long)
        
        self.num_entities = 0
        self.num_relations = 0
        
        if mappings_path and os.path.exists(mappings_path):
            print(f"Loading mappings from {mappings_path}...")
            with open(mappings_path, 'r') as f:
                mappings = json.load(f)
                self.num_entities = len(mappings.get('ent2id', []))
                self.num_relations = len(mappings.get('rel2id', []))
        else:
            print("Mapping file not found. Inferring counts from data (Max ID)...")
            self.num_entities = max(df['h'].max(), df['t'].max()) + 1
            self.num_relations = df['r'].max() + 1
            
        del df 
        print(f"Dataset loaded. Shape: {self.data.shape}")
        print(f"Entities: {self.num_entities}, Relations: {self.num_relations}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        h, r, t = self.data[idx]
        return {
            'head': h,
            'relation': r,
            'tail': t
        }