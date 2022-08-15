Introduction:
This Git repository is the Python implementation of my MSc project *Improve Cold-Start Recommendations with Data Augmentation in Sequential Recommenders*.

The baseline recommender is a self-attentive model called SASRec, with the reference to the original paper attached below.
The authors have published their tensorflow implementation of proposed model in [this repository](https://github.com/kang205/SASRec).
A pytorch version of the code can be found in [this repository](https://github.com/pmixer/SASRec.pytorch).
In this project, the code implementation uses pytorch framework, and has referred to the scripts in the latter repository.

```
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
```

---

Environment:
- CUDA: 10.2
- pytorch: 1.6

---

File Explanation:
- `data/ml-am`: contains the dataset of MovieLens. Some data files are kept in this archive, so the augmentation and model training process can be directly executed.
- `SASRec_pytorch/utils_cs.py`: contains the modified versions of functions from `utils.py`, these functions are used to supports the cold-start training and evaluation.
- `formate_ml-1m.py`: script for processing raw MovieLens-1M datasets (e.g. extract the user behaviour sequences and item list).
- `data_split.py`: used to identify the cold-start users and items, split the sequences in to four splits: ws, ucs, ics, mcs.
- `find_similar.ipynb`: script for finding the synonyms for cold-start items.
- `word2vec`: a folder that contains some training data and stopwords list for Word2Vec and synonym identification.
- `augmentation.py`: the implementation of the data augmentation approaches.
---

Execution:
1. Execute `formate_ml-1m` and `data_split.py` for dataset preprocessing and task sets splitting. 
2. Use `augmentation.py` to apply different data augmentation methods; 
   for the Synonym-Replacement approach, the synonym list should be generated in advance using `find_similar.ipynb`.
3. Run `SASRec.py` for model training and evaluation, the generated augmentation data can be added into the training set by using the command line paramenter `--da_file`
   

Example of applying SynReplace augmentation. 
Assume the dataset has been preprocessed, and the synonyms have been identified. 
The first line of command generates the augmentation data.
The second line of command trains and evaluates the recommendation model. 
The agumentation file is passed into the model via parameter `--da_file`.
The `--cold_start=true` flag is set to activate the cold-start mode (otherwise the mode just works as the original work).
```
python augmentation.py --dataset=ml-1m --method=SynRep --augNum=40 --synonym_file=data/ml-1m/similar_items/wikigiga_100_con
.txt
python SASRec.py --device=cuda --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --lr=0.001 --cold_start=tru
e --da_file=ml-1m.da.SynRep.augNum=40.txt
```

Example execution of SeqSplit augmentation, or mixed SynReplace-SeqSplit augmentation:

```
python augmentation.py --dataset=ml-1m --method=SynRep --augNum=40 --synonym_file_file=data/ml-1m/similar_items/wikigiga_100_con.txt
python SASRec.py --device=cuda --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --lr=0.001 --cold_start=true --da_file=ml-1m.da.SynRep.augNum=40.txt
```

```
python augmentation.py --dataset=ml-1m --method=Mixed --percentage=1.0 --maxAug=3 --synonym_file=data/ml-1m/similar_items/wikigiga_100_con.txt --augNum=40
python SASRec.py --device=cuda --dataset=ml-1m --train_dir=default --cold_start=true --da_file=ml-1m.da.Mixed.SR_N=40.SS_P=1.0.SS_N=3.txt
```