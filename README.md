# SAM
This is a demo fine-tune Segmentanything with Multiple GPUs. This case was trained on three GPUs and use `nn.DataParallel`.
# Installation
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`.

0. Create ./SAM
1. Install [Pytorch](https://pytorch.org/):    
2. Install Segment Anything:    
```
pip install git+https://github.com/facebookresearch/segment-anything.git    
cd SAM
pip install -e . 
```
3. Copy SAM_handover files to SAM
# Checkpoint
Here we take vit_l as an example. You can use vit_B or vit_h according to your personal needs.    
`vit_l` [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)    
`vit_b` [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)    
`vit_h` [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)    
     
# Training    
`python SAM/training.py --batch 32 --dataroot ./trainingdata --model_type vit_l --checkpoint sam_vit_l_0b3195.pth`
