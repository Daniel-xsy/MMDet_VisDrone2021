
The base code is mmdetection 2.12.0

# 2021/6/29
- Annotations:

About sabl_retina_head and bucket_assignment.

- Modify:

Support multi-scale test by ratio;

Example as `img_scale=[2.5, 1.7]` in test config;

The modifiled part is in [test_time_aug.py](./mmdet/datasets/pipelines/test_time_aug.py) line65. `# assert mmcv.is_list_of(self.img_scale, tuple)`


- Add:

Add VisDrone evaluation code.
Related file are [visdrone.py](./mmdet/datasets/visdrone.py) and [viseval.py](./mmdet/datasets/viseval.py)
