from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import numpy as np
import csv
from rdkit.Chem import QED
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw, PandasTools
import sascorer
from sascorer import *
import pandas as pd
from rdkit.Chem import Descriptors
import predefined_models
from RAscore import RAscore_NN

def generate(mol, verbose=False):
    selected_columns = ['nHBAcc', 'nHBDon', 'nRot', 'nBonds', 'nAromBond', 'nBondsO', 'nBondsS',
                        'TopoPSA(NO)', 'TopoPSA', 'LabuteASA', 'bpol', 'nAcid', 'nBase',
                        'ECIndex', 'GGI1', 'SLogP', 'SMR', 'BertzCT', 'BalabanJ', 'Zagreb1',
                        'ABCGG', 'nHRing', 'naHRing', 'NsCH3', 'NaaCH', 'NaaaC', 'NssssC',
                        'SsCH3', 'SdCH2', 'SssCH2', 'StCH', 'SdsCH', 'SaaCH', 'SsssCH', 'SdssC',
                        'SaasC', 'SaaaC', 'SsNH2', 'SssNH', 'StN', 'SdsN', 'SaaN', 'SsssN',
                        'SaasN', 'SsOH', 'SdO', 'SssO', 'SaaO', 'SsF', 'SdsssP', 'SsSH', 'SdS',
                        'SddssS', 'SsCl', 'SsI']

    # Test Data filter
    #test_smiles_list = []
    test_formula_list = []
    test_mordred_descriptors = []


    mol = Chem.AddHs(mol)
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    formula = formula.replace("+", "")
    formula = formula.replace("-", "")

    #test_smiles_list.append(smiles)
    test_formula_list.append(formula)
    test_mordred_descriptors.append(predefined_models.predefined_mordred(mol, "all"))

    # get all column names
    column_names = predefined_models.predefined_mordred(Chem.MolFromSmiles("CC"), "all", True)

    # create Mordred desc dataframe
    test_df = pd.DataFrame(index=test_formula_list, data=test_mordred_descriptors, columns=column_names)

    # Select predefined columns by the model
    selected_data_test = test_df[selected_columns]
    print('selected_data_test:', selected_data_test)
    selected_data_test = selected_data_test.apply(pd.to_numeric)

    return selected_data_test
from streamlit import *
# 数据集包含对NLRP3活性值的判断（1为有活性，0为无活性），QED值和SA_Score值等。
# calculate_qed_sa.py代码为计算QED、SA_Score和MolLogP值，并添加到smiles列后。
# 在存储数据时，在第二列（即smiles后一列）加上1或0来表示有无NLRP3活性。
# 命名为*_qed_sa_LogP.csv（用于第一步训练Transformer）或*_for_trainRNN.csv（用于第三步训练RNN）

df = pd.read_csv('')
# df = df[df['SPLIT'] == 'train']
datanames = df['SMILES']
print('datanames:', datanames)
nn_scorer = RAscore_NN.RAScorerNN()
for smi in datanames:
    # img = Draw.MolToImage(mol, size=(300, 300))
    if smi is None:
        pass
    else:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            pass
        else:
            qed = '%.3f' % QED.qed(mol)
            print('qed:', qed)
            MolLogP = '%.3f' % Descriptors.MolLogP(mol)
            print('MolLogP:', MolLogP)
            SA_Score = '%.3f' % my_score(mol)
            with open("", "a") as f:  # a:循环写入；w:非循环写入
                f.write(
                    '\n' + str(smi) +',' +'1'+ ',' + str(qed) + ',' + str(SA_Score) + ','+ str(
                        MolLogP) )  # 0：无活性；1：有活性






