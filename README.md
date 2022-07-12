Introduction:
This Git repository is the Python implementation of my MSc project *Improving Cold-Start Ability of Sequential Recommenders using Data Augmentation*.

The baseline recommender is a self-attentive model called SASRec, with the reference to the original paper attached below.
The Authors have published their tensorflow implementation of proposed model in [this repository](https://github.com/kang205/SASRec).
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

Version:
- CUDA: 10.2
- pytorch: 1.6

---

Script:
- `formate_ml-1m.py`: process raw datasets (e.g. extract the user behaviour sequences and item list).
- `data_split.py`: identify the cold-start users and items, split the sequences in to four splits: ws, ucs, ics, mcs.
- `utils_cs.py`: contains the modified versions of functions from `utils.py`, these functions are used to supports the cold-start training and evaluation.
---

Execution:
1. Execute `formate_ml-1m` and `data_split.py` for dataset preprocessing and splitting. 
2. Use `augmentation.py` to apply different data augmentation methods.
3. Run `SASRec.py` for training and evaluation on the SASRec model, use `--cold_start=true` flag to activate the cold-start mode (otherwise the mode just works as original).
E.g. Train a SASRec model with a specified data augmentation file:

E.g. Run a pretrained model, evaluate its performance only:
```
python SASRec.py --device=cuda --dataset=ml-1m --train_dir=default --state_dict_path='ml-1m_default/SASRec.epoch=601.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' --inference_only=true --maxlen=200

```

