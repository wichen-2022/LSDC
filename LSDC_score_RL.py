#!/usr/bin/env python
from __future__ import print_function, division
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs
import rdkit.Chem.QED as QED
from utils import sascorer
import pickle
from focal_loss import BinaryFocalLoss
from tensorflow.keras.models import load_model
from rdkit.Chem import Descriptors
rdBase.DisableLog('rdApp.error')



class tanimoto():
    """Scores structures based on Tanimoto similarity to a query structure.
       Scores are only scaled up to k=(0,1), after which no more reward is given."""

    kwargs = ["k", "query_structure"]
    k = 0.7
    query_structure = " "

    def __init__(self):
        query_mol = Chem.MolFromSmiles(self.query_structure)
        self.query_fp = AllChem.GetMorganFingerprint(query_mol, 2, useCounts=True, useFeatures=True)

    def __call__(self, smiles_list):
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                pass
            else:
                fp = AllChem.GetMorganFingerprint(mol, 2, useCounts=True, useFeatures=True)
                score = DataStructs.TanimotoSimilarity(self.query_fp, fp)
                score = min(score, self.k) / self.k
        return float(score)
        return 0.0

class NLRP3_model():


    def __init__(self):
        kwargs = ["mlp_path"]
        mlp_path = 'data/nlrp3/model_9-07-0.998501.hdf5'
        self.mlp = load_model(mlp_path) # custom_objecta={'loss': binary_crossentropy}

    def __call__(self, smiles_list):
        fps = []
        mask = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append(int(mol is not None))
            fp = np.array(AllChem.RDKFingerprint(mol, fpSize=2048)).reshape(1, 2048) if mol else np.zeros((1, 2048))
            fps.append(fp)
        fps = np.concatenate(fps, axis=0)



        scores = self.mlp.predict(fps)  # scores = self.mlp.predict(fps)[:, 1]
        #scores = [1 if val >= 0.8 else 0 if val <= 0.2 else 0.5 for val in scores]
        scores = [1 if val > 0.5 else 0  for val in scores]
        scores = scores * np.array(mask)

        return np.float32(scores)

    #@classmethod
    #def transfer_label_from_prob(preLabel_probality):
    #    preLabel_twoClass = [1 if val >= 0.8 else -1 if val <= 0.2 else 0 for val in preLabel_probality]
    #    return preLabel_twoClass

class qed_func():

    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(0)
            else:
                try:
                    qed = QED.qed(mol)
                except:
                    qed = 0
                scores.append(qed)
        return np.float32(scores)


class sa_func():

    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(100)
            else:
                scores.append(sascorer.calculateScore(mol))
                #print('sa scores:', scores)
        return np.float32(scores)

class logP_func():

    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(0)
            else:
                try:
                    logP =  Descriptors.MolLogP(mol)
                except:
                    logP = 0
                scores.append(logP)
        return np.float32(scores)



def get_scoring_function(prop_name):
    """Function that initializes and returns a scoring function by name"""

    if prop_name == 'qed':
        return qed_func()
    elif prop_name == 'sa':
        return sa_func()
    elif prop_name == 'logP':
        return logP_func()
    elif prop_name == 'NLRP3':
        return NLRP3_model()

    # else:
    #     return chemprop_model(prop_name)


def multi_scoring_functions(data, function_list):
    funcs = [get_scoring_function(prop) for prop in function_list]
    props = np.array([func(data) for func in funcs])

    scoring_sum = props.sum(axis=0)

    return scoring_sum


def multi_scoring_functions_one_hot(data, function_list):
    funcs = [get_scoring_function(prop) for prop in function_list]
    props = np.array([func(data) for func in funcs])

    props = pd.DataFrame(props).T
    props.columns = function_list

    scoring_sum = condition_convert(props).values.sum(1)

    # scoring_sum = props.sum(axis=0)

    return scoring_sum


def condition_convert(con_df):
    # convert to 0, 1

    con_df['NLRP3'][con_df['NLRP3'] >= 0.5] = 1
    con_df['NLRP3'][con_df['NLRP3'] < 0.5] = 0
    con_df['qed'][con_df['qed'] >= 0.6] = 1
    con_df['qed'][con_df['qed'] < 0.6] = 0
    con_df['sa'][con_df['sa'] <= 4.0] = 1
    con_df['sa'][con_df['sa'] > 4.0] = 0
    con_df['logP'][con_df['logP'] <= 5.0] = 1
    con_df['logP'][con_df['logP'] > 5.0] = 0
    return con_df


if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--prop', required=True)

    args = parser.parse_args()
    funcs = [get_scoring_function(prop) for prop in args.prop.split(',')]

    data = [line.split()[:2] for line in sys.stdin]
    all_x, all_y = zip(*data)
    props = [func(all_y) for func in funcs]

    col_list = [all_x, all_y] + props
    for tup in zip(*col_list):
        print(*tup)
