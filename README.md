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

Versions:
- CUDA: 10.2
- pytorch: 1.6

---

Scripts:
- `cold_start.py`: the main script for functionalities 
- `utils_cs.py`: contains the modified versions of functions from `utils.py`, these functions are used to supports the cold-start training and evaluation.
---

Execution:

Execute `cold_start.py` for dataset splitting, preprocessing and data augmentation; then run the `main.py` for recommender training and evaluation.


E.g. Run a pretrained model, evaluate its performance only:
```
python main.py --device=cuda --dataset=ml-1m --train_dir=default --state_dict_path='ml-1m_default/SASRec.epoch=601.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' --inference_only=true --maxlen=200

```

