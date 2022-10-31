# Obj-SA-GAN
## Obj-SA-GAN - PyTorch Implementation

<img src="overall.png"/>

### Dependencies
python 3.6

Pytorch 0.4.1

In addition, please add the project folder to PYTHONPATH and `pip install` the following packages:
- `python-dateutil`
- `easydict`
- `pandas`
- `torchfile`
- `nltk`
- `scikit-image`
- `spacy`
- `PyYAML`
- `cffi`
- `torchtext`
- `dill`
- `Cython`

========================= Datasets, pre-training models, etc. are consistent with Obj-GAN =========================
**Data**

1. Download our preprocessed metadata for [coco](https://drive.google.com/open?id=1GbZESaDwkpV8gH2gyo1bUogPtYu1QEPF) and merge them to `data/coco`
2. Download [coco](http://cocodataset.org/#download) dataset, extract the images to `data/coco/images`, and extract the annotations to `data/coco/insanns`

**Training**

- Train box generator:
  - `cd box_generation`
  - `python sample.py --is_training 1`
- Train shape generator:
  - `cd shape_generation`
  - `./make.sh`
  - `python main.py --gpu '0,1' --FLAG`
- Train image generator:
  - `cd image_generation`
  - `./make.sh`
  - `python main.py --gpu '0,1' --FLAG`

**Pretrained Model**

Download and save them to `data/coco/pretrained/`
- [DAMSM for coco](https://drive.google.com/open?id=1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ)
- [Inception v3](https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth)
- [VGG19 BN](https://download.pytorch.org/models/vgg19_bn-c79401a0.pth)
- [Box generator](https://drive.google.com/file/d/1OTZDywt1UGzUykAXBXmvVA6aAlQzbMjv/view?usp=sharing)
- [Shape generator](https://drive.google.com/file/d/1vyfXxh4eC1ccs9XNhC8OIylErhwLdvmN/view?usp=sharing)
- [Image generator](https://drive.google.com/file/d/1BWXJT5Wg0x0Ajatgb2VdSQG14ndG8CGM/view?usp=sharing)
