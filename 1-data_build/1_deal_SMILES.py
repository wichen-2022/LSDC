from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolStandardize
from rdkit.Chem.MolStandardize import rdMolStandardize

# 数据清洗：用于建模前处理分子smiles，可以得到去除盐或络合物，去除部分金属原子，中和分子的数据。
# 但实际操作发现，数据不能完全清洗干净，会留下部分含有盐或者小片段的smiles，此时需要在excel表格中筛选smiles列，选择含有.的行进行删除即可。

lfc = MolStandardize.fragment.LargestFragmentChooser()  # 只留下最大片段，去除盐或络合物
md = MolStandardize.rdMolStandardize.MetalDisconnector()  # 去金属原子
Uncharger= rdMolStandardize.Uncharger()  # 中和分子：Uncharger考虑整个分子的电中性，如果整个分子是电中性的，即使原子带电荷也不会进行处理

data = pd.read_csv('')
smis = data['SMILES']
nlrp3 = data['NLRP3']  # 获取NLRP3列数据
print('smis:', smis)
smi2s = []
#remover = SaltRemover(defnData="[Cl,Br]")
for smi in smis:
    mol = Chem.MolFromSmiles(smi)
    #mol2 = remover.StripMol(mol)
    if mol is not None:
        mol2 = lfc.choose(mol)
        print('mol2:', mol2)
        mol3 = md.Disconnect(mol2)
        print('mol3:', mol3)
        mol4 = Uncharger.uncharge(mol3)
        print('mol4:', mol4)

        smi2 = Chem.MolToSmiles(mol4)
        print(smi + "->" + smi2)
        smi2s.append(smi2)
# 创建包含拼接NLRP3列的DataFrame
stan_smiles = pd.DataFrame(smi2s, columns=['SMILES'])
stan_smiles['NLRP3'] = nlrp3
print('stan_smiles:', stan_smiles)
stan_smiles.to_csv('', index=False)