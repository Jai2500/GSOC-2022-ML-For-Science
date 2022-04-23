# GSOC 2022 Tasks for ML for Science 

**Name: Jai Bardhan**

Models developed and trained on PyTorch. Experiment Logging was implemented using Weights and Biases.

## Task 1

Electron vs Photon classifier: Achieves required AUROC. Implemented with Pytorch and uses pretrained models.

### How to run

1. Download the required dataset.
2. Change the path in the `Task1/train.py` to the path of the downloaded dataset
3. Run `python Task1/train.py`


## Task 2

Quark vs Gluon Classifier: Achieves good AUROC. Implemented with PyTorch and PyArrow for dataset. Uses a pretrained Resnet-50. However larger showed better performance but could not be trained due to compute constraints.

### How to run

1. Download the required dataset.
2. Change the path in the `Task2/train.py` to the path of the downloaded dataset
3. Run `python Task2/train.py`

To run the **regress** task. 

Run `python Task2/regress.py` instead.

## Task 3

GNN Classifier:

### Simple GAT

The dataset was constructed by choosing columns in the detectors that had non zero on any of the channel.

A simple Graph Attention Network was implemented. The reason for choosing attention over edges, was that the edges were made with `k=20` nearest neighbours and it is very likely that some edges do not contribute. 
An attention layer would allow the network to reason the weights (or _attention_) over the nearest neighbours.

#### How to run
1. Download the required dataset.
2. Change the `model` paramter to `gat` in the `Task3/train.py` file. 
3. Change the path in the `Task3/train.py` to the path of the downloaded dataset
4. Run `python Task3/train.py`

### ParticleNet DGCNN

Another idea would be to allow the network to itself find the underlying data manifold and find relevant nodes and make edges with those. Thus a ParticleNet style DGCNN was implemented. 
The `pos` was the position of the column on the detectors, and the `x` was the values + pos of the columns of the detector.

#### How to run
1. Download the required dataset.
2. Change the `model` paramter to `dgcnn` in the `Task3/train.py` file. 
3. Change the path in the `Task3/train.py` to the path of the downloaded dataset
4. Run `python Task3/train.py`

