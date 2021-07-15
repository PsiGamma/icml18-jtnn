import torch
from torch.utils.data import Dataset
from mol_tree import MolTree
import numpy as np
from jtnn.mpn import mol2graph

class MoleculeDataset(Dataset):

    def __init__(self, data_file):
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        smiles = self.data[idx]
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()
        return mol_tree

class PropDataset(Dataset):
    def __init__(self, data_file, prop_file):
        self.prop_data = np.loadtxt(prop_file)
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        smiles = self.data[idx]
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()
        return mol_tree, self.prop_data[idx]

import pandas as pd;
class ToxPropDataset(Dataset):
    def __init__(self, smiles: pd.DataFrame, targets: pd.DataFrame):
        self.targets = targets;
        self.data = smiles;
        self.list_IDs = smiles.index;

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        # Select sample
        structure_id = self.list_IDs[index]        
        smile = self.data.loc[structure_id]
        
        mol_tree = MolTree(smile)
        mol_tree.recover()
        mol_tree.assemble()
        
        if self.targets is not None:
            y = torch.from_numpy(
                np.array(self.targets.loc[structure_id])
            ).float()
        else:
            y = torch.tensor(-1).float()
            
        return mol_tree, y


class ToxMultiPropDataset(Dataset):
    def __init__(self, smiles: pd.DataFrame, targets: pd.DataFrame, out_size: int):
        self.targets = targets;
        self.data = smiles;
        self.list_IDs = smiles.index;
        self.out_size = out_size 

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        # Select sample
        structure_id = self.list_IDs[index]        
        smile = self.data.loc[structure_id]
        
        mol_tree = MolTree(smile)
        mol_tree.recover()
        mol_tree.assemble()
        
        if self.targets is not None:
            y = torch.from_numpy(
                np.array(self.targets.loc[structure_id])
            ).float()
        else:
            y = torch.ones(out_size).float()
            
        return mol_tree, y
    

class SmilesPropDataset(Dataset):
    def __init__(self, list_IDs, smiles, prop):
        self.list_IDs = list_IDs;
        self.prop_data = prop;
        self.data = smiles;

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]        
        smile = self.data[ID]
        return smile, torch.from_numpy(np.array(self.prop_data[ID])).float();
    
