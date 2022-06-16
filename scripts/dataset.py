import os, pickle
import pandas as pd

from rdkit import Chem

import torch
import sklearn
import dgl.backend as F
from dgl.data.utils import save_graphs, load_graphs

def match_atomidx(rsmi, psmi):
    rmol = Chem.MolFromSmiles(rsmi)
    pmol = Chem.MolFromSmiles(psmi)
    atom_map2idx = {}
    atom_p2r = {}
    for atom in rmol.GetAtoms():
        if atom.GetSymbol() == 'H':
            continue
        atom_idx = atom.GetIdx()
        atom_map = atom.GetAtomMapNum()
        atom_map2idx[atom_map] = atom_idx
    for atom in pmol.GetAtoms():
        if atom.GetSymbol() == 'H':
            continue
        atom_idx = atom.GetIdx()
        atom_map = atom.GetAtomMapNum()
        atom_p2r[atom_idx] = atom_map2idx[atom_map]
    return atom_p2r

class DiffDataset(object):
    def __init__(self, split, smiles_to_graph, node_featurizer, edge_featurizer, load=True, log_every=1000):
        df = pd.read_csv('../data/%s.csv' % split)
        self.rsmiles = df['rsmi'].tolist()
        self.psmiles = df['psmi'].tolist()
        self.labels = df['ea'].tolist()
        self.rgraphs_path = '../data/saved_graphs/%s_rgraphs.bin' % split
        self.pgraphs_path = '../data/saved_graphs/%s_pgraphs.bin' % split
        self.p2rs_path = '../data/saved_graphs/%s_p2rs.pkl' % split
        self._pre_process(split, smiles_to_graph, node_featurizer, edge_featurizer, load, log_every)

    def _pre_process(self, split, smiles_to_graph, node_featurizer, edge_featurizer, load, log_every):
        if os.path.exists(self.rgraphs_path) and load:
            print('Loading previously saved %s graphs...' % split)
            self.rgraphs, _ = load_graphs(self.rgraphs_path)
            self.pgraphs, _ = load_graphs(self.pgraphs_path)
            with open(self.p2rs_path, 'rb') as f:
                self.atom_p2rs = pickle.load(f)
        else:
            print('Processing dgl graphs from scratch...')
            self.rgraphs = []
            self.pgraphs = []
            self.atom_p2rs = []
            for i, (rsmi, psmi) in enumerate(zip(self.rsmiles, self.psmiles)):
                if (i + 1) % log_every == 0:
                    print('\rProcessing molecule %d/%d' % (i+1, len(self.rsmiles)), end='', flush=True)
                self.rgraphs.append(smiles_to_graph(rsmi, node_featurizer=node_featurizer,
                                                   edge_featurizer=edge_featurizer, canonical_atom_order=False))
                self.pgraphs.append(smiles_to_graph(psmi, node_featurizer=node_featurizer,
                                                   edge_featurizer=edge_featurizer, canonical_atom_order=False))
                self.atom_p2rs.append(match_atomidx(rsmi, psmi))
            print ()
        save_graphs(self.rgraphs_path, self.rgraphs)
        save_graphs(self.pgraphs_path, self.pgraphs)
        with open(self.p2rs_path, 'wb') as f:
            pickle.dump(self.atom_p2rs, f)
        
    def __getitem__(self, item):
        return self.rgraphs[item], self.pgraphs[item], self.atom_p2rs[item], self.labels[item]

    def __len__(self):
            return len(self.rsmiles)