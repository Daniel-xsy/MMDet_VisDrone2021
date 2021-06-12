from .builder import DATASETS
from .coco import CocoDataset


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