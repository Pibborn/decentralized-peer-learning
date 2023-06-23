# Peer Learning: *Learning Complex Policies in Groups from Scratch via Action Recommendations*

Multiple agents (*peers*) learning together simultaneously from scratch with the ability to communicate in a '*What would you do in my situation?*' manner.

## Installation
All packages are specified for Python version 3.7.4.

```bash
pip install -r requirements.txt
pip install stable-baselines3==1.5.0 --no-dependencies
```

## Sample Usage

The entry point to Peer Learning is the ``run_peer.py`` file.
You can simply do Peer Learning with the default setting by running the following line of code:

```bash
python run_peer.py
```

## Experiments

In [Experiments.md](Experiments.md), you find the commands to replicate the experiments reported in our paper including all hyperparameters.
