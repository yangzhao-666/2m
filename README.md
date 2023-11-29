# Two-Memory Reinforcement Learning
IEEE Conference on Games 2023

[Zhao Yang](https://yangzhao-666.github.io), [Thomas Moerland](https://thomasmoerland.nl), [Mike Preuss](https://scholar.google.se/citations?user=KGlyGUcAAAAJ&hl=en), [Aske Plaat](https://askeplaat.wordpress.com)

<img src="https://github.com/yangzhao-666/TwoM/blob/main/2M.png" width="600" height="300">

---

# Codebase still WIP!

To learn more:
- [paper](https://arxiv.org/abs/2304.10098)
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

### Running Experiments
--- 
You can simply run the code ```python train_2m.py```, or you can change hyper-parameters in ```train_2m.py``` then run it.

Please be noted hyper-parameters in this work is quite senstive, in order to fully reproduce the results presented in the paper, you need to set values for all hyper-parameters as follows:

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
```
### Acknowledgements
--- 
2M builds on many prior works, and we thank the authors for their contributions.
- [MinAtar](https://github.com/kenjyoung/MinAtar/tree/master) for simplified Atari tasks
- [MFEC](https://github.com/astier/model-free-episodic-control/tree/master) for the implementation of model-free episodic control agent
- [PEG](https://github.com/penn-pal-lab/peg/tree/master) for their nice READMEs
