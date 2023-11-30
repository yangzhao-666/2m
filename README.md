# Two-Memory Reinforcement Learning
IEEE Conference on Games 2023

[Zhao Yang](https://yangzhao-666.github.io), [Thomas Moerland](https://thomasmoerland.nl), [Mike Preuss](https://scholar.google.se/citations?user=KGlyGUcAAAAJ&hl=en), [Aske Plaat](https://askeplaat.wordpress.com)

<img src="https://github.com/yangzhao-666/TwoM/blob/main/2M.png" width="600" height="300">

---
To learn more:
- [paper](https://arxiv.org/abs/2304.10098)

If you find our paper or code useful, please reference us:

```
@article{yang2023two,
  title={Two-Memory Reinforcement Learning},
  author={Yang, Zhao and Moerland, Thomas and Preuss, Mike and Plaat, Aske},
  booktitle={IEEE Conference on Games},
  year={2023}
}
```

### Dependencies Installation
Create the conda environment by running:
```
conda env create -f environment.yml
```

In order to run experiments on [MinAtar](https://github.com/kenjyoung/MinAtar) tasks, you need to install MinAtar correctly by following instructions provided.

### Running Experiments
--- 
The code base uses [wandb](https://wandb.ai) for logging all the results, for using it, you need to register as a user. Then you can pass ```--wandb``` to enable wandb logging.

You can simply run the code ```python train_2m.py --wandb```, tabular experiments presented in the paper ```python tabular/train_tab.py --wandb```.

Please be noted hyper-parameters in this work are quite senstive, in order to fully reproduce the results presented in the paper, you need to set hyper-parameters the same as in [file](https://github.com/yangzhao-666/2m/blob/main/hyper_setting.json).


### Code Overview
---
The structure of the code base.
```
2m/
  |- train_2m.py            # start training
  |- DQN.py                 # implementation of DQN agent
  |- MFEC_atari.py          # implementation of model-free episodic control agent for MinAtar tasks
  |- tabular/               # folder of tabular implementations
  |- RB.py                  # implementation of replay buffers
  |- utils.py               # utils functions
```
### Acknowledgements
--- 
2M builds on many prior works, and we thank the authors for their contributions.
- [MinAtar](https://github.com/kenjyoung/MinAtar/tree/master) for simplified Atari tasks
- [MFEC](https://github.com/astier/model-free-episodic-control/tree/master) for the implementation of model-free episodic control agent
- [PEG](https://github.com/penn-pal-lab/peg/tree/master) for their nice READMEs
