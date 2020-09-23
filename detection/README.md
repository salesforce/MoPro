## Finetuning MoPro pre-trained model for object detection on COCO

1. Install <a href="https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md">detectron2</a>.
2. Convert model into detectron 2 format
<pre>
python convert-pretrain-to-detectron2.py [pre-trained model path] mopro.pkl
</pre>
3. Put dataset under "./datasets" directory, following the directory structure requried by <a href="https://github.com/facebookresearch/detectron2/tree/master/datasets">detectron2</a>.
4. Run training:
<pre>
python train_net.py --config-file configs/coco_R_50_FPN_1x.yaml \
 --num-gpus 8 MODEL.WEIGHTS ./mopro.pkl
 </pre>
 
## Results using Faster R-CNN with a R50-FPN backbone, 1x schedule: 
Pre-train dataset| AP | AP50 | AP75
 --- | --- | --- | ---
WebVision V1 | 39.7 | 60.9 | 43.1
WebVision V2 | 40.1 | 61.7 | 44.5
