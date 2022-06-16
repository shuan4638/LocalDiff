import json
import pandas as pd
from rdkit import Chem
import torch
from torch import nn
import sklearn

from rdkit.Chem import PandasTools, AllChem

import dgl
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from functools import partial

from scripts.utils import init_featurizer, load_model, collate_molgraphs, predict

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

def init_LocalDiff(args):
    args['mode'] = 'test'
    args = init_featurizer(args)
    model = load_model(args)
    smiles_to_graph = partial(smiles_to_bigraph, add_self_loop=True)
    node_featurizer = CanonicalAtomFeaturizer()
    edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
    graph_function = lambda s: smiles_to_graph(s, node_featurizer = node_featurizer, edge_featurizer = edge_featurizer, canonical_atom_order = False)
    return model, graph_function
    
def get_Ea(rsmiles, psmiles, args, model, graph_function):
    model.eval()
    rgraph, pgraph = graph_function(rsmiles), graph_function(psmiles)
    p2r = match_atomidx(rsmiles, psmiles)
    with torch.no_grad():
        output = predict(args, model, rgraph, pgraph, p2r)

    return output.item()
