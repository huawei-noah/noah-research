## Conversation Graph: Data Augmentation, Training and Evaluation for Non-Deterministic Dialogue Management

Hello :) This is the accompanying code for our [TACL paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00352/97777/Conversation-Graph-Data-Augmentation-Training-and) about data augmentation, training and evaluation for non-deterministic dialogue management. Any questions? Email milan-dot-gritta-at-huawei-dot-com. If you found this resource useful, please cite the paper as follows:

```
@article{10.1162/tacl_a_00352,
    author = {Gritta, Milan and Lampouras, Gerasimos and Iacobacci, Ignacio},
    title = "{Conversation Graph: Data Augmentation, Training, and Evaluation for Non-Deterministic Dialogue Management}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {9},
    pages = {36-52},
    year = {2021},
    month = {02},
    abstract = "{Task-oriented dialogue systems typically rely on large amounts of high-quality training data or require complex handcrafted rules. However, existing datasets are often limited in size con- sidering the complexity of the dialogues. Additionally, conventional training signal in- ference is not suitable for non-deterministic agent behavior, namely, considering multiple actions as valid in identical dialogue states. We propose the Conversation Graph (ConvGraph), a graph-based representation of dialogues that can be exploited for data augmentation, multi- reference training and evaluation of non- deterministic agents. ConvGraph generates novel dialogue paths to augment data volume and diversity. Intrinsic and extrinsic evaluation across three datasets shows that data augmentation and/or multi-reference training with ConvGraph can improve dialogue success rates by up to 6.4\\%.}",
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00352},
    url = {https://doi.org/10.1162/tacl\_a\_00352},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00352/1923936/tacl\_a\_00352.pdf},
}
```

#### Getting Started

- Download **MultiWOZ** from the [ConvLab](https://github.com/ConvLab/ConvLab/tree/master/data/multiwoz) website (required files: test.json.zip, val.json.zip, train.json.zip) and place them (unzipped) into the `multiwoz` folder
- Download the **M2M/Self-Play** (train.json, dev.json, test.json) datasets for [restaurant](https://github.com/google-research-datasets/simulated-dialogue/tree/master/sim-R) and [movie](https://github.com/google-research-datasets/simulated-dialogue/tree/master/sim-M) and place them into the `self_play/restaurant` and `self_play/movie` folders, respectively.
- Create a conda environment with **conda create -n convgraph python==3.7**
- Activate it with **conda activate convgraph**
- Install pytorch with something like **conda install pytorch==1.7.0 torchvision==0.8.1 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch**
- Then do **pip install -r requirements.txt** to get the rest of the packages...
- You're good to go! :)

#### Running Code

- You can run all experiments for MultiWOZ from within `multiwoz/evaluation.py`
- For Machines2Machines data, use `self_play/evaluation.py`
- You can adjust the hyperparameters at the top of each evaluation file
  - For example, you can change `history`, `dataset`, `train_with_soft_loss`, `max_epochs`, `max_val_f1`, `patience`
  - The default values used in the paper are given but feel free to tune further
- For different baselines and setups, uncomment the code inside each comment block, for example... 
- To run MFS only:
```
# data augmentation training only
x_train, y_train = train_graph.generate_augmented_data()
```
- To run the base model:
```
# baseline training
x_train, y_train = train_graph.generate_standard_data(unique=False)
```
- To run the MFS+Baseline model:
```
# data augmentation + baseline training
x_t, y_t = train_graph.generate_augmented_data()
x_train, y_train = train_graph.generate_standard_data(unique=False)
x_train = np.concatenate((x_train, x_t))
y_train = np.concatenate((y_train, y_t))
```  

- We hope the comments in each file will be self-explanatory
- Run experiments with several seeds to obtain a stable average
- Let us know if you have any issues by reaching out to milan-dot-gritta-at-huawei-dot-com

##### Paper Links

- [MIT Press](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00352/97777/Conversation-Graph-Data-Augmentation-Training-and) for the journal publication, presented at [ACL 2021](https://aclanthology.org/2021.tacl-1.3/) :)

##### Extrinsic Evaluation

- We used [ConvLab 1.0](https://github.com/ConvLab/ConvLab) to evaluate the models using their [user simulator](https://github.com/ConvLab/ConvLab#evaluation)
- Warning: It is not the easiest software to use, be patient
- If you have trouble with running extrinsic evaluation, email my colleague gerasimos-dot-lampouras-at-huawei-dot-com :)
