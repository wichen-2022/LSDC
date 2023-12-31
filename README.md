#  LSDC

## 1. Data processing and data partitioning

### (1) deal_SMILES.py 

Firstly, use the deal_SMILES code to process the initial data set.

### (2) data_structs.py 

The data_structs code is used at this time to continue cleaning the data. 

The voc character library generated at this time cannot be used.

### (3) calculate_qed_sa.py 

At this time, the calculate_qed_sa code is used to calculate qed, sa, logP, etc. 
After getting test.csv, you also need to calculate attributes such as qed, sa, logP and add them to the data set.

### (4) 4_bulid_voc 

Build a character library and then divide the train and valid data sets.

## 2. Prior_train.py     

Use the positive and negative samples of NLRP3 to train the pre-trained model.

## 3. Generator_train.py

Targeting NLRP3 to generate a large number of compounds using trained generative models.

## 4. Dm_train.py

Use the compounds generated in the previous step as an expanded training set to train the distillation model.

## 5. LSDC-agent_train.py

Use LSDC's reinforcement learning strategy to train the agent model to improve the skeleton diversity of generated molecules.

## References
<a id="1">[1]</a> 
Jike Wang et al.
Multi-constraint molecular generation based on conditional transformer, knowledge distillation and reinforcement learning. 
Nature Machine Intelligence, 3, 914–922 (2021).
