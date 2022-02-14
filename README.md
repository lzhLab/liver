This study present an elaborated large-volume and high-quality dataset containing liver raw images, liver masks and liver vessel masks; in addition, a bi-directional scaling algorithm having shallow down-scaling and deep up-scaling is proposed for fine-grained vessel segmentation.

# Dataset

The masked labels as well as the raw images created in this study, namely LVSD300, has been uploaded to the IEEE DataPort with DOI:10.21227/rwys-mk84 (URL: https://ieee-dataport.org/documents/liver-vessel).

# Usage

### Parameters

* `num_workers`: int
   <br>Number of workers. Used to set the number of threads to load data.
* `ckpt`: str
  <br>Weight path. Used to set the dir path to save model weight. 
* `w`: str
  <br>The path of model wight to test or reload.
* `heads`: int
  <br>Number of heads in Multi-head Attention layer.
* `mlp_dim`: int.
  <br>Dimension of the MLP (FeedForward) layer.
* `channels`: int, default 3.
  <br>Number of image's channels.
* `dim`: int.
  <br>Last dimension of output tensor after linear transformation `nn.Linear(..., dim)`.
* `dropout`: float between `[0, 1]`, default 0.
  <br>Dropout rate.
* `emb_dropout`: float between `[0, 1]`, default 0.
  <br>Embedding dropout rate.
* `patch_h` and `patch_w`:int
  <br>The patches size.
* `dataset_path`: str
  <br>Used to set the relative path of training and validation set.
* `batch_size`: int
  <br>Batch size.
* `max_epoch`: int 
  <br>The maximum number of epoch for the current training.
* `lr`: float
  <br>learning rate. Used to set the initial learning rate of the model.
