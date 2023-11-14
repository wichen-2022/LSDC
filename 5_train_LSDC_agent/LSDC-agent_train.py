#!/usr/bin/env python
import argparse
import warnings

import torch

import numpy as np
import pandas as pd
import time

from models.model_rnn import RNN
from utils.data_structs import Vocabulary, Experience
from utils.LSDC_score_RL import get_scoring_function, qed_func, sa_func,logP_func,multi_scoring_functions_one_hot
from utils.utils import Variable, seq_to_smiles, fraction_valid_smiles, unique
from reinvent_scoring.scoring.diversity_filters.reinvent_core.diversity_filter_memory import *
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import *
from reinvent_scoring.scoring.diversity_filters.reinvent_core.scaffold_similarity import *
warnings.filterwarnings("ignore")
import os
from rdkit.Chem import AllChem
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def scaffold_function(smile):

    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        pass
    else:
        standardized_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        # IsomericSmiles 参数设置为 True，这意味着 SMILES 字符串中包含立体化学信息。
        # 规范参数设置为 True，这意味着将以规范化形式生成 SMILES 字符串。
        # 这确保了化学等效分子将具有相同的 SMILES 字符串，而不管它们在输入摩尔对象中的具体表示方式如何。
        # print(standardized_smiles)

        Pred_retain_scaffold = []
        No_center_nitrogen_smiles = []
        Unexpected_number_fragments_smiles = []
        mol = Chem.MolFromSmiles(standardized_smiles)
        smi_scaffolds = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        # print('smi_scaffolds:', smi_scaffolds)
        mol_scaffolds = Chem.MolFromSmiles(smi_scaffolds)

        # 获取所有原子的序号和符号
        atom_symbols = [atom.GetSymbol() for atom in mol_scaffolds.GetAtoms()]
        atom_indices = [atom.GetIdx() for atom in mol_scaffolds.GetAtoms()]
        # 寻找中心氮原子
        center_nitrogen_idx = -1
        for atom in mol_scaffolds.GetAtoms():
            if atom.GetSymbol() == 'N' and len(atom.GetNeighbors()) == 2 and not atom.IsInRing():
                center_nitrogen_idx = atom.GetIdx()
                break

        if center_nitrogen_idx == -1:
            #print("No center nitrogen atom found in molecule:", standardized_smiles)
            retain_scaffold = None
            No_center_nitrogen_smiles.append(smiles)
            with open(" ", "w") as file:
               for element in No_center_nitrogen_smiles:
                   file.write(element + "\n")
        else:
            # 分割成两个片段
            bond_indices = [bond.GetIdx() for bond in mol_scaffolds.GetBonds() if
                            bond.GetBeginAtomIdx() == center_nitrogen_idx]
            mol_frag = Chem.FragmentOnBonds(mol_scaffolds, bond_indices, addDummies=False)

            mol_frags = Chem.GetMolFrags(mol_frag, asMols=True)
            if len(mol_frags) != 2:
                #print("Unexpected number of fragments found in molecule:", standardized_smiles)
                retain_scaffold = None
                # Unexpected_number_fragments_smiles.append(smiles)
                # with open("RESULT/Unexpected_number_fragments_smiles.csv", "w") as file:
                #   for element in Unexpected_number_fragments_smiles:
                #      file.write(element + "\n")
            else:
                fragment1 = Chem.MolToSmiles(mol_frags[0], isomericSmiles=False, canonical=True)
                fragment2 = Chem.MolToSmiles(mol_frags[1], isomericSmiles=False, canonical=True)
                retain_scaffold = choose_mol(fragment1, fragment2)
                #print('Pred_retain_scaffold:', Pred_retain_scaffold)
        return retain_scaffold

def choose_mol(mol1, mol2): # customized
    if len(mol1) >= len(mol2):
        return mol1
    else:
        return mol2

def _add_to_memory_dataframe(self, step: int, smile: str, scaffold: str, component_scores: Dict):
    data = []
    headers = []
    for name, score in component_scores.items():
        headers.append(name)
        data.append(score)
    headers.append("Step")
    data.append(step)
    headers.append("Scaffold")
    data.append(scaffold)
    headers.append("SMILES")
    data.append(smile)
    new_data = pd.DataFrame([data], columns=headers)
    self._memory_dataframe = pd.concat([self._memory_dataframe, new_data], ignore_index=True, sort=False)


def train_agent(restore_prior_from='',
                restore_agent_from='',
                agent_save='',
                batch_size=128, n_steps=60, sigma=60, save_dir='',
                experience_replay=0):
    voc = Vocabulary(init_from_file="Voc")

    start_time = time.time()

    Prior = RNN(voc)
    Agent = RNN(voc)

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(restore_prior_from, map_location={'cuda:0': 'cuda:0'}))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from))
    else:
        Prior.rnn.load_state_dict(torch.load(restore_prior_from, map_location=lambda storage, loc: storage))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

    # We dont need gradients with respect to Prior
    for param in Prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=0.0001)

    experience = Experience(voc)
    #print('experience:', experience)

    print("Model initialized, starting training...")

    # Scoring_function
    scoring_function1 = get_scoring_function('NLRP3')  # NLRP3
    smiles_save = []
    expericence_step_index = []
    my_scaffold_1 = []
    my_scaffold = []
    for step in range(n_steps):

        # Sample from Agent
        seqs, agent_likelihood, entropy = Agent.sample(batch_size=batch_size)

        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # Get prior likelihood and score
        prior_likelihood = Prior.likelihood(Variable(seqs))
        smiles = seq_to_smiles(seqs, voc)
        #print('smiles:', smiles)

        score1 = scoring_function1(smiles)

        qed = qed_func()(smiles)
        sa = np.array([float(x < 4.0) for x in sa_func()(smiles)],
                      dtype=np.float32)  # to keep all reward components between [0,1]
        # for循环迭代出 logP_func(smiles) 中的每个值，并将其与 5.0 进行比较。
        # 对于小于5.0的情况，用 float(x < 5.0) 将True转换为1.0，将False转换为0.0，生成一个新的列表
        logP = np.array([float(x < 5.0) for x in logP_func()(smiles)],
                      dtype=np.float32)

        score = np.array(score1) + np.array(qed) + np.array(sa) +np.array(logP)

        # 判断是否为success分子，并储存
        success_score = multi_scoring_functions_one_hot(smiles, ['NLRP3', 'qed', 'sa', 'logP'])
        #print('success_score:', success_score)
        itemindex = list(np.where(success_score == 4))
        success_smiles = np.array(smiles)[itemindex]

        index = pd.RangeIndex(start=0, stop=len(set(success_smiles)))
        my_ScaffoldSimilarity = ScaffoldSimilarity(BaseDiversityFilter)
        my_memory = DiversityFilterMemory()
        bucket_size = 10
        True_retain_smiles = pd.read_csv('', header=None).values.reshape(-1)
        True_retain_smiles = scaffold_function(True_retain_smiles)
        True_mols = [Chem.MolFromSmiles(s) for s in True_retain_smiles]
        True_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in True_mols]

        smiles_remove_unmatched = []
        for i in index:
            smi = success_smiles[i]
            if "." in smi:
                fragments = smi.split('.')
                longest_fragment = max(fragments, key=len)
                # 将处理后的分子转换成SMILES字符串输出
                smiles_without_unmapped_atoms = longest_fragment
                smiles_remove_unmatched.append(smiles_without_unmapped_atoms)
            else:
                smiles_remove_unmatched.append(smi)

        for smile in smiles_remove_unmatched:
            scaffold1 = scaffold_function(smile)
            #scaffold = my_ScaffoldSimilarity._find_similar_scaffold(scaffold1)
            if scaffold1 is None:    # 除去scaffold为none的分子
                scores = 0
            else:
                Pred_mol = Chem.MolFromSmiles(scaffold1)
                Pred_fp = AllChem.GetMorganFingerprintAsBitVect(Pred_mol, 3, 2048)
                Similarity = DataStructs.BulkTanimotoSimilarity(Pred_fp, True_fps)
                if max(Similarity) < 0.5:   # 除去与高活专利库骨架最高相似性大于等于0.5的分子
                    scores = 4
                    my_scaffold_1.append(scaffold1)
                    scaffold_count = np.sum(
                        np.array((pd.DataFrame(my_scaffold_1).values == scaffold1)).astype(int))
                    if scaffold_count < bucket_size:  # 除去与已生成分子骨架同类超过bucket_size的分子
                        #print('my_smile:', smile)
                        smiles_save.append(smile)
                        #print('my_scaffold:', scaffold1)
                        my_scaffold.append(scaffold1)

        # 构建 DataFrame 对象
        df = pd.DataFrame({"smiles": smiles_save})
        # 将 DataFrame 存储到 csv 文件中
        df.to_csv(save_dir + '', index=False)

        df = pd.DataFrame({"my_scaffold": my_scaffold})
        df.to_csv(save_dir + '', index=False)

        expericence_step_index = expericence_step_index + len(success_smiles) * [step]
        print('step:', step)
        print('n_steps:', n_steps)

        #if step >= n_steps:
        print('num of smiles_save: ', len(set(smiles_save)))

        torch.save(Agent.rnn.state_dict(), agent_save)

        # Calculate augmented likelihood
        augmented_likelihood = prior_likelihood + sigma * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Experience Replay
        # First sample
        if experience_replay and len(experience) > 4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(4)
            exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        # Then add new experience
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, score, prior_likelihood)
        experience.add_experience(new_experience)

        # Calculate loss
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        loss_p = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p
        print('loss:', loss)

        # Calculate gradients and make an update to the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        print("\n       Step {}   Fraction valid SMILES: {:4.1f}  Time elapsed: {:.2f}h Time left: {:.2f}h".format(
            step, fraction_valid_smiles(smiles) * 100, time_elapsed, time_left))
        print("  Agent    Prior   Target   Score             SMILES")
        for i in range(10):
            print('i:', i)
            print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i],
                                                                       prior_likelihood[i],
                                                                       augmented_likelihood[i],
                                                                       score[i],
                                                                       smiles[i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for running the model")
    parser.add_argument('--num-steps', action='store', dest='n_steps', type=int,
                        default=60)
    parser.add_argument('--batch-size', action='store', dest='batch_size', type=int,
                        default=128)
    parser.add_argument('--sigma', action='store', dest='sigma', type=int,
                        default=60)
    parser.add_argument('--middle', action='store', dest='restore_prior_from',
                        default='',
                        help='Path to an RNN checkpoint file to use as a Prior')
    parser.add_argument('--agent', action='store', dest='agent_save',
                        default='',
                        help='Path to an RNN checkpoint file to use as a Agent.')
    parser.add_argument('--save-file-path', action='store', dest='save_dir',
                        default='',
                        help='Path where results and model are saved. Default is data/results/run_<datetime>.')

    arg_dict = vars(parser.parse_args())

    train_agent(restore_prior_from='',
                restore_agent_from='',
                agent_save='',
                batch_size=128, n_steps=60, sigma=60, save_dir='',
                experience_replay=0)
