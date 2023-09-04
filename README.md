# Triple-CMR

A Deep Neural Network model accepting three modality-data as input. It is proposed as a solution to the problem of information retrieval in the context of three robotic sensory inputs: visual, auditory, and tactile.

Visual and tactile data is represented by RGB images, while audio samples are saved as .wav files. 

The first stage of training the model involves training with cross-entropy loss. The three modalities have separate pathways in the early layers: Resnet-50 for Images and 1d-CNN for Audio. Then, two modalities, which make up retrieval space are fused using multihead attention mechanism.

A final stage of training is extracting feature embeddings from last hidden layers of separate pathways, and using triplet loss, further bringing closer representations of the same objects in latent space.

The evaluation of cross-retrieval performance is done using Mean Average Precision (MAP) metric.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Command-line Arguments](#command-line-arguments)
- [Experimental Results](#experimental-results)
  - [Dataset](#Dataset)
  - [Results](#Results)
- [License](#license)

## Installation

Clone the repository and install required prerequisites. 
Project was developed in Python 3.8.10, requirements.txt is provided.

```bash
git clone https://github.com/jagodawojcik/Triple-CMR.git
cd yourrepository
pip install -r requirements.txt
```

## Usage

Here's how to run the Triple-CMR model training with specifying minimum required parameters:

```bash
python main.py --query-modality QUERY_MODALITY --dominating-modality DOMINATING_MODALITY
```

### Command-line Arguments

#### `--query-modality`

*Required* Set the query modality:

- `visual`
- `audio`
- `tactile`

#### `--dominating-modality`

*Required* Set the dominating modality, representation of which will be used to compute cross-entropy loss. Joint embedding is a concatenated representation of all three modalities from the last layer of joint network.

- `visual`
- `audio`
- `tactile`
- `joint_embedding`

Following parameters, can also be specified, but are not required, default values are provided.

#### `--epoch-pretrain`

Set the number of epochs for tactile branch pretraining Default: `15`

#### `--epoch-c-entropy`

Set the number of epochs for the training with cross-entropy loss. Default: `50`

#### `--batch-size-c-entropy`

Set the batch size for the cross-entropy training stage - both pretrain and the main training stage. Default: `5`

#### `--epoch-triplet`

Set the max range of epochs for triplet loss training. Default: `19500`

#### `--batch-size-triplet`

Set the batch size for the triplet loss training stage. Default: `5`

#### `--margin-triplet`

Set the margin size for triplet loss optimization. Default: `0.5`

#### `--use-linux-echo`

Set logging type as True for Linux echo print or False for standard Python print. Default: `False`

## Experimental Results

### Dataset
Link to dataset: [here](https://drive.google.com/drive/folders/1tUKbRt5QgVkjYPtqwllNBinv4hjf9rzW?usp=drive_link).
Dataset used in a set of experiments conducted in Triple-CMR study consists of visual, audio and tactile representations of 20 objects, has the following distribution.

| Data Type       | Number of Images |
|-----------------|------------------|
| Total           | 34,500           |
*From which*:
| Training        | 25,500           |
| Validation      | 4,500            |
| Test & Evaluation | 4,500          |

Dataset was generated from ObjectFolder 2.0 Dataset created by R. Gao et al., code available here https://github.com/rhgao/ObjectFolder.

### Results

Include results.

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file in the repository.
