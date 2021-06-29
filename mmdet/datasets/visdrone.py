import mmcv
import time
import numpy as np
from collections import OrderedDict
from mmcv.utils import print_log

from .builder import DATASETS
from .coco import CocoDataset
from .viseval import VisDroneEval


@DATASETS.register_module()
class VisdroneDataset(CocoDataset):
	CLASSES = ('pedestrian', 'people', 'bicycle','car', 
		'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
	def __init__(self, out_annots=False, *args, **kwargs):
		super(VisdroneDataset, self).__init__(*args, **kwargs)
		self.out_annots = out_annots


	def get_subset_by_classes(self):
		"""Get img ids that contain any category in class_ids.

		Different from the coco.getImgIds(), this function returns the id if
		the img contains one of the categories rather than all.

		Args:
			class_ids (list[int]): list of category ids

		Return:
			ids (list[int]): integer list of img ids
		"""
		if self.test_mode:
			return self.data_infos
		ids = set()
		for i, class_id in enumerate(self.cat_ids):
			ids |= set(self.coco.cat_img_map[class_id])
		self.img_ids = list(ids)

		data_infos = []
		for i in self.img_ids:
			info = self.coco.load_imgs([i])[0]
			info['filename'] = info['file_name']
			data_infos.append(info)
		return data_infos

	def __getitem__(self, idx):
		if self.test_mode and not self.out_annots:
		    return self.prepare_test_img(idx)
		while True:
			data = self.prepare_train_img(idx)
			if data is None:
				idx = self._rand_another(idx)
				continue
			return data

	def format_results_for_evaluate(self, results):
		'''
		input_file format:
		mmdet.detection_task.pkl
		[img_num,class_num,prediction_of_per_class_num]
		prediction_of_per_class_num[0] = [x1, y1, x2, y2, score]
		
		return:
		list[predictions_results_for_each_img]
		prediction_results[0] = [x1, y1, w, h, score, cat_id]
		sort in descending order of scores
		'''
		all_dt = []
		st = time.time()
		assert len(results) == len(self.coco.imgs)
		for img_predicts in results:
			det_results = []
			for cls_index in range(len(img_predicts)):
				each_cls_predicts = img_predicts[cls_index]
				each_cls_predicts = np.concatenate(
					[each_cls_predicts,
					cls_index*np.ones((each_cls_predicts.shape[0],1))],
					axis=1)
				w=each_cls_predicts[:,2] - each_cls_predicts[:,0]
				h=each_cls_predicts[:,3] - each_cls_predicts[:,1]
				each_cls_predicts[:,2] = w
				each_cls_predicts[:,3] = h
				det_results.append(each_cls_predicts)
			det_results=np.concatenate(det_results,axis=0)
			order = np.argsort(-det_results[:,4])
			det_results = det_results[order,:]
			
			all_dt.append(det_results)

		print('Done (t=%.2fs).'%(time.time()-st))
		return all_dt
	
	def format_coco_for_evaluate(self):
		'''convert coco format to 
		list[gts for each img]
		gts[0] = [x1, y1, w, h, score, cat_id]
		score is iscrowded.

		img_size format: list[list[w,h]]
		'''
		img_size = []
		all_gt = []
		for i in self.coco.getImgIds():
			img_info = self.coco.loadImgs(i)[0]
			img_size.append([img_info['width'], img_info['height']])
			annots_info = self.coco.loadAnns(self.coco.getAnnIds(i))
			bboxes = np.array([x['bbox'] for x in annots_info])
			cats = np.array([x['category_id'] for x in annots_info])
			scores = 1 - np.array([x['iscrowd'] for x in annots_info])
			temp = np.concatenate((bboxes, scores[:,None], cats[:,None]),axis=1)
			all_gt.append(temp)
		return all_gt, img_size

	def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 max_dets=[1, 10, 100, 500],
                 iou_thrs=None,
                 metric_items=None):
		'''Evaluation in VisDrone protocol.

		Args:
			results (list[list | tuple]): Testing results of the dataset.
			metric (str | list[str]): Metrics to be evaluated. Options are
				only 'bbox'.
			logger (logging.Logger | str | None): Logger used for printing
				related information during evaluation. Default: None.
			max_dets (Sequence[int]): Proposal number used for evaluating
				recalls, such as recall@10, recall@500.
				Default: [1, 10, 100, 500].
			iou_thrs (Sequence[float], optional): IoU threshold used for
				evaluating recalls/mAPs. If set to a list, the average of all
				IoUs will also be computed. If not specified, [0.50, 0.55,
				0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
				Default: None.
			metric_items (list[str] | str, optional): Metric items that will
				be returned. If not specified, ``['mAP', 'mAP_50', 'mAP_75',
				'mAR_1', 'mAR_10', 'mAR_100', 'mAR_500']`` will be used when
				``metric=='bbox'.

		Returns:
			dict[str, float]: according to metric items.
			'''

		metrics = metric if isinstance(metric, list) else [metric]
		allowed_metrics = ['bbox']
		for metric in metrics:
			if metric not in allowed_metrics:
				raise KeyError(f'metric {metric} is not supported')
		if iou_thrs is None:
			iou_thrs = np.linspace(
				.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
		if metric_items is None:
			metric_items = [
				'mAP', 'mAP_50', 'mAP_75', 'mAR_1', 'mAR_10', 'mAR_100', 'mAR_500'
			]
		elif not isinstance(metric_items, list):
			metric_items = [metric_items]

		all_dt = self.format_results_for_evaluate(results)
		all_gt, img_size = self.format_coco_for_evaluate()

		eval_results = OrderedDict()
		for metric in metrics:
			msg = f'Evaluating {metric}...'
			if logger is None:
				msg = '\n' + msg
			print_log(msg, logger=logger)
		

		eval_class = self.coco.loadCats(self.coco.get_cat_ids())
		visEval = VisDroneEval(all_dt, all_gt, img_size, eval_class, iou_thrs=iou_thrs, max_dets=max_dets)
		visEval.evaluate()
		eval_results = visEval.summarize()

		return eval_results