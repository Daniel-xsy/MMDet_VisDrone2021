import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class VisDroneEval(object):
    def __init__(self, all_dt, all_gt, img_size, eval_class, iou_thrs=None, max_dets=None):
        self.all_dt = all_dt
        self.all_gt = all_gt
        self.img_size = img_size
        self.eval_class = eval_class
        self.iou_thrs = iou_thrs
        self.max_dets = max_dets
        if iou_thrs is None:
            self.iou_thrs = np.arange(0.5, 1, 0.05)
        if max_dets is None:
            self.max_dets = [1, 10, 100, 500]

        self.cls_num = len(self.eval_class)
        self.max_num = len(self.max_dets)
        self.iou_num = len(self.iou_thrs)
        self.AP = np.zeros((self.cls_num, self.iou_num))
        self.AR = np.zeros((self.cls_num, self.iou_num, self.max_num))

        self.eval_class_num = [0]*self.cls_num

    def cal_ioa(self,boxes,ignores,w,h,intermap=None):
        '''
		intersection over self_area:inter/(h*w).
		'''
        assert boxes.shape[1] == ignores.shape[1] ==4
        boxes = np.maximum(1, np.round(boxes)).astype(np.int32)
		
        area = boxes[:,2]*boxes[:,3]

        boxes[:,2] += boxes[:,0]
        boxes[:,3] += boxes[:,1]
        boxes[:,[0,2]] = np.clip(boxes[:,[0,2]],a_min=1,a_max=w)
        boxes[:,[1,3]] = np.clip(boxes[:,[1,3]],a_min=1,a_max=h)
		
        if intermap is None:
            ##generate ignore region intergral map
            ignores = np.maximum(1,np.round(ignores)).astype(np.int32)
            ignores[:,2] += ignores[:,0]
            ignores[:,3] += ignores[:,1]
            ignores[:,[0,2]] = np.clip(ignores[:,[0,2]],a_min=1,a_max=w)
            ignores[:,[1,3]] = np.clip(ignores[:,[1,3]],a_min=1,a_max=h)
            
            ign_map = np.zeros((h+1,w+1))
            for x1,y1,x2,y2 in ignores:
                ign_map[y1:y2+1,x1:x2+1] = 1
            img_integral = cv2.integral(ign_map)
        else:
            img_integral = intermap

        ###to align matlab code results
        boxes += 1

        tl = img_integral[boxes[:,1],boxes[:,0]]
        tr = img_integral[boxes[:,1],boxes[:,2]]
        bl = img_integral[boxes[:,3],boxes[:,0]]
        br = img_integral[boxes[:,3],boxes[:,2]]
        inter_area = (tl+br-tr-bl)

        return inter_area/area, img_integral


    def drop_objects_in_ingore(self):
        '''
        drop annotations and detections in ignored region.

        If the overlap between one box and ignored region is more than
        half of the box area, drop it.
        '''
        st = time.time()
        print('Drop ignore region detection results...')

        for idx,(w,h) in enumerate(self.img_size):

            old_gt = self.all_gt[idx]
            old_dt = self.all_dt[idx]
            ##transform to numpy format
            old_gt = old_gt if isinstance(old_gt, np.ndarray) else np.array(old_gt)
            old_dt = old_dt if isinstance(old_dt, np.ndarray) else np.array(old_dt)

            cur_gt = old_gt[old_gt[:,5]!=-1] #category == -1 means ignore
            ign_gt = old_gt[old_gt[:,5]==-1]

            if ign_gt.shape[0] > 0:
                ioa,intermap = self.cal_ioa(cur_gt[:,:4].copy(),ign_gt[:,:4].copy(),w,h)
                cur_gt = cur_gt[ioa<0.5]
                ioa,_ = self.cal_ioa(old_dt[:,:4].copy(),ign_gt[:,:4].copy(),w,h,intermap)
                old_dt = old_dt[ioa<0.5]

            self.all_gt[idx] = cur_gt
            self.all_dt[idx] = old_dt

        print('Done (t=%.2fs).'%(time.time()-st))


    def evaluate(self, show=False):

        ##drop GT and detections in ignores regions
        self.drop_objects_in_ingore()

        st = time.time()
        for idx,cat_dict in enumerate(self.eval_class):
            key = cat_dict['id']
            value = cat_dict['name']
            print('evaluating object category:', value, '[%d/%d]...'%(idx+1, self.cls_num))
            gt_matches, dt_matches = self.eval_each_class(idx, key)
            self.accumulate(gt_matches, dt_matches, idx, show)
        print('Done (t=%.2fs)'%(time.time()-st))

    def calc_oas(self, dt, gt, ig_id):
        assert dt.shape[1]==gt.shape[1]==4

        dt_area = dt[:,2]*dt[:,3]
        gt_area = gt[:,2]*gt[:,3]

        ##transform to [x1, y1, x2, y2]
        dt[:,2] += dt[:,0]
        dt[:,3] += dt[:,1]
        gt[:,2] += gt[:,0]
        gt[:,3] += gt[:,1]

        iw = np.minimum(gt[:,None,2],dt[:,2]) - np.maximum(gt[:,None,0],dt[:,0])
        ih = np.minimum(gt[:,None,3],dt[:,3]) - np.maximum(gt[:,None,1],dt[:,1])
        iw[iw<=0] = 0
        ih[ih<=0] = 0
        intersection = iw*ih

        ua = gt_area[:,None]+dt_area-intersection
        ##ignore 
        ua[ig_id] = dt_area

        return intersection / ua

    def eval_each_class(self, n, cls_id):
        '''
        gt_line=[x1, y1, w, h, score, cat_id]
        dt_line=[x1, y1, w, h, score, cat_id]
        '''
        gt_matches = defaultdict(lambda: [[] for i in range(self.max_num)])
        dt_matches = defaultdict(lambda: [[] for i in range(self.max_num)])
        for idx in range(len(self.img_size)):
            gt_all = self.all_gt[idx].copy()
            dt_all = self.all_dt[idx].copy()
            
            gt_all = gt_all[gt_all[:,5]==cls_id]
            gt_all[gt_all[:,4]==0,4] = 1 ##gt ignore
            gt_all[gt_all[:,4]==1,4] = 0
            ##sort gt ignore last
            order = np.argsort(gt_all[:,4])
            gt_all = gt_all[order,:]
            ##gt_flg{'ignore':-1, 'matched':1, 'unmatched':0}
            gt_all[:,4] = -gt_all[:,4]
            ig_id = (gt_all[:,4]==-1)
            ig_or = (ig_id.sum()>0)
            
            if gt_all.shape[0]>0:
                self.eval_class_num[n] += 1

            for i, max_det in enumerate(self.max_dets):
                dt = dt_all[:max_det,:].copy()
                dt = dt[dt[:,5]==cls_id,]
                gt = gt_all.copy()

                ##sort dt highest score first
                order = np.argsort(-dt[:,4])
                dt = dt[order,:]
                ##dt_flg{'matched':1, 'unmatched':0,'matched_ign':-1}
                dt[:,5] = 0
                ##when gt or dt is empty
                if gt.shape[0]<=0 or dt.shape[0]<=0:
                    for thr in self.iou_thrs:
                        gt_matches[thr][i].append(gt[:,4].copy())
                        dt_matches[thr][i].append(dt[:,4:].copy())
                    continue

                ##begin calclrate match between gt and dt
                oa = self.calc_oas(dt[:,:4], gt[:,:4], ig_id)
                
                for thr in self.iou_thrs:
                    iou = oa.copy()
                    dt_temp = dt.copy()
                    gt_temp = gt.copy()
                    if ig_or:
                        iou = iou[~ig_id]
                        iou_ig = iou[ig_id]

                    for dt_count in range(dt.shape[0]):
                        max_id = np.argmax(iou[:,dt_count],axis=0)
                        max_iou = iou[max_id,dt_count]
                        if max_iou>=thr:
                            # dt_temp[dt_count,4] = max_iou
                            dt_temp[dt_count,5] = 1
                            gt_temp[max_id,4] = 1
                            iou[max_id,:] = 0
                        elif ig_or:
                            max_id = np.argmax(iou_ig[:,dt_count],axis=0)
                            max_iou = iou_ig[max_id,dt_count]
                            if max_iou>=thr:
                                dt_temp[dt_count,5] = -1
                    gt_matches[thr][i].append(gt_temp[:,4])
                    dt_matches[thr][i].append(dt_temp[:,4:])

        return gt_matches, dt_matches

    
    def voc_ap(self, recall, precision, thr, show=False, ax=None):
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        ##show AP-curse
        if show:
            ax.plot(mrec[:-1], mpre[:-1], color="r", linestyle="-", linewidth=1)
            ax.annotate("%.2f"%thr, xy=(mrec[-10],mpre[-10]))
        return ap


    def accumulate(self, gt_matches, dt_matches, cls_num, show=False):
        
        ax = None
        if show:
            fig, ax = plt.subplots(1,1)
            plt.axis([0, 1, 0, 1])
            plt.title(self.eval_class[list(self.eval_class.keys())[cls_num]])

        for i, thr in enumerate(self.iou_thrs):
            for j in range(self.max_num):
                gt_match = np.concatenate(gt_matches[thr][j],0)
                dt_match = np.concatenate(dt_matches[thr][j],0)

                order = np.argsort(-dt_match[:,0])
                dt_match = dt_match[order,:]
                tp = np.cumsum(dt_match[:,1]==1)
                rec = tp/max(1,len(gt_match))
                if rec.shape[0]> 0:
                    self.AR[cls_num, i, j]=max(rec)*100

            fp = np.cumsum(dt_match[:,1]==0)
            prec = tp/np.maximum(1,(fp+tp))
            self.AP[cls_num, i] = self.voc_ap(rec, prec, thr, show, ax=ax)*100
        if show:
            plt.show()
        

    def summarize(self):
        self.eval_class_num = np.array(self.eval_class_num)
        self.eval_class_num = self.eval_class_num/self.eval_class_num.sum()
        
        ap = self.AP.mean(1)
        print()
        for i, cat_dict in enumerate(self.eval_class):
            value = cat_dict['name']
            print('%s (AP) @[ IoU=0.50:0.95 | maxDets=500 ] = %.2f%%'%(value+' '*(17-len(value)), ap[i]))
        
        print()
        ap_all = (self.eval_class_num*ap).sum()
        print('Average Precision (AP) @[ IoU=0.50:0.95 | maxDets=500 ] = %.2f%%'%(ap_all))
        ap_50 = (self.eval_class_num*self.AP[:,0]).sum()
        print('Average Precision (AP) @[ IoU=0.50      | maxDets=500 ] = %.2f%%'%(ap_50))
        ap_75 = (self.eval_class_num*self.AP[:,5]).sum()
        print('Average Precision (AP) @[ IoU=0.75      | maxDets=500 ] = %.2f%%'%(ap_75))

        ar = self.AR.mean(1)
        ar = (self.eval_class_num[:,None]*ar).sum(0)
        print()
        for i, value in enumerate(self.max_dets):
            value = str(value)
            print('Average Recall    (AR) @[ IoU=0.50:0.95 | maxDets=%s ] = %.2f%%'%(value+' '*(3-len(value)), ar[i]))

        return {
            'AP':[float(f'{x:.3f}') for x in ap],
            'mAP':float(f'{ap_all:.3f}'),
            'mAP_50':float(f'{ap_50:.3f}'),
            'mAP_75':float(f'{ap_75:.3f}'),
            'mAR_1':float(f'{ar[0]:.3f}'),
            'mAR_10':float(f'{ar[1]:.3f}'),
            'mAR_100':float(f'{ar[2]:.3f}'),
            'mAR_500':float(f'{ar[3]:.3f}'),
        }