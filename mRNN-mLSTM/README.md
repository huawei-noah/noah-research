# Memory-augmented Recurrent Networks (mRNN-mLSTM)

### Paper

Do RNN and LSTM have Long Memory? ICML 2020. [[paper]](https://proceedings.icml.cc/static/paper_files/icml/2020/956-Paper.pdf)

By Jingyu Zhao, Feiqing Huang, Jia Lv, Yanjie Duan, Zhen Qin, Guodong Li and Guangjian Tian.

Please refer to the paper for an introduction to datasets and the required references.

### Requirements

- python 3
- pytorch >= 1.0.0
- Gensim
- sklearn
- numpy
- json
- pandas
- math

### Usage

#### Time series prediction

  For example, you can run the following code to train an `mLSTM` model on the `tree7` dataset.

```
  python train.py --dataset 'tree7' --algorithm 'mLSTM'
```

**Available datasets**

- ARFIMA series: `arfima`
- Dow Jones Industrial Average (DJI): `DJI`
- Metro interstate traffic volume: `traffic`
- Tree ring: `tree7`

**Available algorithms**

- vanilla RNN: `RNN`
- vanilla LSTM: `LSTM`
- Memory-augmented RNN with homogeneous memory parameter d: `mRNN_fixD`
- Memory-augmented RNN with dynamic d: `mRNN`
- Memory-augmented LSTM with homogeneous d: `mLSTM_fixD`
- Memory-augmented LSTM: `mLSTM`

#### Review classification

- You can use word2vec to embed each word to a vector with Gensim. The parameter `vec_size` is used for setting the embedding dimension.

  ```
  python preprocess.py --vec_size 16
  ```

  Here, we have already embedded each word to a 16-dimension vector and saved the embeddings in `data.json` in `data\review_classification`. 

- You can select and test different models through the `algorithm` parameter. For example, you can test the model `mLSTM_fixD` using the code below.

  ```
  python train.py --algorithm 'mLSTM_fixD'
  ```

**Available algorithms**

- vanilla RNN: `RNN`
- vanilla LSTM: `LSTM`
- Memory-augmented RNN with homogeneous memory parameter d: `mRNN_fixD`
- Memory-augmented LSTM with homogeneous d: `mLSTM_fixD`

