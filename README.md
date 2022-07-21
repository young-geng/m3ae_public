# Multimodal Masked Autoencoders (M3AE): A JAX/Flax Implementation

This is a JAX/Flax re-implementation for the paper [Multimodal Masked Autoencoders Learn Transferable Representations](https://arxiv.org/abs/2205.14204).

```
@article{geng2022multimodal,
  title={Multimodal Masked Autoencoders Learn Transferable Representations},
  author={Geng, Xinyang and Liu, Hao and Lee, Lisa and Schuurams, Dale and Levine, Sergey and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2205.14204},
  year={2022}
}
```

This implementation has been tested on GPU and Google Cloud TPU and supports multi-host training with TPU Pods.
Unliked the original implementation used for the paper, this implementation also supports the following new
features:
* Predicting discretized image tokens from VQGAN as output (similar to BEiT).
* Training on a combination of paired image-text data and unpaired text data.

## Installation
Install the dependencies with pip and add this repo directory to your`PYTHONPATH` environment variable.
```
pip install requirements.txt
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```


## Running Experiments
Experiments can be launched via the following commands.

Pre-training MAE (image only model) on CC12M
```
python3 -m m3ae.mae_main \
    --mae.model_type='large' \
    --mae.use_type_embedding=False \
    --seed=42 \
    --epochs=100 \
    --lr_warmup_epochs=5 \
    --batch_size=4096 \
    --dataloader_n_workers=16 \
    --log_freq=500 \
    --plot_freq=2000 \
    --save_model_freq=10000 \
    --lr_peak_value=1.5e-4 \
    --weight_decay=0.05 \
    --discretized_image=False \
    --load_checkpoint='' \
    --dataset='cc12m' \
    --cc12m_data.path="<YOUR DATA HDF5 FILE PATH>" \
    --cc12m_data.image_normalization='cc12m'
```

Pre-training M3AE (image and text model) on CC12M
```
python3 -m m3ae.m3ae_main \
    --m3ae.model_type='large' \
    --m3ae.image_mask_ratio=0.75 \
    --m3ae.text_mask_ratio=0.75 \
    --seed=42 \
    --epochs=100 \
    --lr_warmup_epochs=5 \
    --batch_size=4096 \
    --discretized_image=False \
    --dataloader_n_workers=16 \
    --log_freq=500 \
    --plot_freq=2000 \
    --save_model_freq=10000 \
    --image_loss_weight=1.0 \
    --text_loss_weight=0.5 \
    --lr_peak_value=1.5e-4 \
    --weight_decay=0.05 \
    --load_checkpoint='' \
    --data.path="<YOUR DATA HDF5 FILE PATH>" \
    --data.transform_type='pretrain' \
    --data.image_normalization='cc12m'
```

Linear classification on ImageNet for both pre-trained MAE and M3AE
```
python3 -m m3ae.linear_main \
    --mae.model_type="large" \
    --mae.use_type_embedding=True \
    --seed=42 \
    --epochs=90 \
    --batch_size=2048 \
    --lr_warmup_epochs=10 \
    --discretized_image=False \
    --dataloader_n_workers=16 \
    --dataloader_shuffle=False \
    --log_freq=500 \
    --save_model_freq=10000 \
    --lr_peak_value=1e-1 \
    --weight_decay=0 \
    --momentum=0.9 \
    --train_data.partition="train" \
    --val_data.partition="val" \
    --train_data.path="<YOUR DATA HDF5 FILE PATH>" \
    --val_data.path="<YOUR DATA HDF5 FILE PATH>" \
    --train_data.transform_type="linear_prob" \
    --val_data.transform_type="test" \
    --load_checkpoint='' \
    --load_pretrained="<YOUR PRE-TRAINED MODEL PATH>"
```

Finetuning on ImageNet for both pre-trained MAE and M3AE
```
python3 -m m3ae.finetune_main \
    --seed=42 \
    --mae.model_type=large \
    --mae.drop_path=0.1 \
    --weight_decay=0.05 \
    --mixup_alpha=0.8 \
    --cutmix_alpha=1.0 \
    --switch_prob=0.5 \
    --label_smoothing=0.1 \
    --layer_decay=0.60 \
    --clip_gradient=1e9 \
    --batch_size=1024 \
    --warmup_epochs=5 \
    --epochs=100 \
    --dataloader_n_workers=16 \
    --dataloader_shuffle=False \
    --log_freq=500 \
    --save_model_freq=10000 \
    --lr_peak_value=1e-3 \
    --train_data.partition="train" \
    --val_data.partition="val" \
    --train_data.path="<YOUR DATA HDF5 FILE PATH>" \
    --val_data.path="<YOUR DATA HDF5 FILE PATH>" \
    --train_data.transform_type="finetune" \
    --val_data.transform_type="test" \
    --load_pretrained="<YOUR PRE-TRAINED MODEL PATH>"
```

## HDF5 Data Format
In order to facilitate training on cloud, we store all the dataset
as HDF5 files and read them from cloud storage buckets. For paired image and text
dataset, the HDF5 data contains two field, `jpg` and `caption`. The `jpg` field
is an 1D array containing the raw bytes of JPEG encoded images. The `caption`
field is an 1D array of utf-8 encoded text. For ImageNet dataset, the image JPEG
bytes are stored in field `train_jpg` and `val_jpg`, and the integer labels are
stored in field `train_labels` and `val_labels`. For unpaired text only dataset,
the utf-8 encoded text is stored in field `text`.


## Credits
* The MAE is heavily inspired by the [original MAE implementation](https://github.com/facebookresearch/mae).

* The VQGAN image tokenizers are from [MaskGiT](https://github.com/google-research/maskgit)
and [dalle-mini](https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384).

* The transformer implementation is heavily inspired by [jax-models](https://github.com/DarshanDeshpande/jax-models).

* Some utilities are borrowed from [JaxCQL](https://github.com/young-geng/JaxCQL).