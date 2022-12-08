

The official code of paper [CEP3: Community Event Prediction with Neural Point Process on Graph](https://openreview.net/forum?id=sfc0rjCBqS_), which is accepted by [Learning on Graph conference](https://logconference.org/).




Please first install MPI on your machine:
<https://www.open-mpi.org/software/ompi/v4.1/>

To run our experiment please install:
```
pip install torch dgl scikit-learn scipy mpi4py
```
or see the requirements.txt 

Our MPI implementation uses toolchain provided by OpenAI

<https://github.com/openai/spinningup>


### Using wikipedia or mooc dataset
As long as the data folder has been created, the dataset will be download automatically if using wikipedia or mooc dataset.
The Github and SocialEvolve datasets are provided as DGLgraph .bin file.

### Use Pretrained Model

```
python train.py --dataset github --use-savedmodel <Model Name>
```

### Train from scratch

#### On Github dataset

```
python train.py --dataset github/wikipedia/social/mooc
```

#### Parallel Parallel Training
```
python train_mpi.py --dataset <mooc/github/wikipedia/social_evolve> --cpus <num_cpus>
```


