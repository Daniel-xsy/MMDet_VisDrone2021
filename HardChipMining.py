import os
import cv2 as cv
import random

anno_path = 'annotations'
imgs_path = 'images'
hc_anno_path = 'hard_chips_annotations'

category = {0 : 'ignored', 1 : 'pedestrian', 2 : 'people', 3 : 'bicycle', 4 : 'car', 
            5 : 'van', 6 : 'truck', 7 : 'tricycle', 8 : 'awning_tri', 9 : 'bus',
            10 : 'motor', 11 : 'others'}
gt_per_cate_per_img = [0] * 10

count = 0

def object_pool_gen():
    for file_name in os.listdir(anno_path):
        anno_file = anno_path + '/' + file_name
        imgs_file = imgs_path + '/' + file_name.split('.')[0] + '.jpg'

        with open(anno_file, 'r') as f:
            bboxes = f.readlines()
            src_img = cv.imread(imgs_file)
            for gt in bboxes:
                gt.strip('\n')

                left, top, width, height, _, cate_id = tuple([int(gt.split(',')[i]) for i in range(6)])
                if cate_id == 0 or cate_id == 11:
                    continue
                gt_per_cate_per_img[cate_id-1] += 1

                gt_img = src_img[top:top+height, left:left+width]
                if height * width < 25:
                    continue
                if os.path.exists(category[cate_id]+'/'+file_name.split('.')[0]+'_'+ str(gt_per_cate_per_img[cate_id-1]) +'.jpg'):
                    continue 
                #print(str(gt_per_cate_per_img[cate_id-1]) + '_')
                cv.imwrite(category[cate_id]+'/'+file_name.split('.')[0]+'_'+ str(gt_per_cate_per_img[cate_id-1]) +'.jpg',
                            gt_img)
        print(imgs_file + 'Completed!')
        for idx in range(10):
            gt_per_cate_per_img[idx] = 0
    


def overlapped_with_gt(l, t, w, h, existed_gt):
    #patch_center = (l + w / 2, t + h / 2)
    
    for gt in existed_gt:
        left_column_max  = max(l, gt[0])
        right_column_min = min(l + w, gt[0] + gt[2])
        up_row_max       = max(t, gt[1])
        down_row_min     = min(t + h, gt[1] + gt[3])
        #print(gt)
        if left_column_max>=right_column_min or down_row_min<=up_row_max:
            continue    #IOU = 0
        else:
            S1 = w * h
            S2 = gt[2] * gt[3]
            S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
            if S_cross/(S1+S2-S_cross) > 0.2 or S1 <= S_cross or S2 <= S_cross:
                return True

    # for gt in existed_gt:
    #     if patch_center[0] > gt[0] and patch_center[0] < gt[0] + gt[2] and patch_center[1] > gt[1] and patch_center[1] < gt[1] + gt[3]:
    #         return True
    return False



def patch_aug():
    srcimg_cate = []
    count = 0
    for file_name in os.listdir(imgs_path):
        print(count)
        count += 1

        img_file = imgs_path + '/' + file_name
        src_img = cv.imread(img_file)
        anno_file = open(anno_path + '/' + file_name.split('.')[0] + '.txt', 'r')
        gt_boxes = anno_file.readlines()
        existed_gt = []
        if len(gt_boxes) > 20:
            continue
        
        for gt in gt_boxes:
            gt_left, gt_top, gt_width, gt_height, _, cate_id = tuple([int(gt.split(',')[i]) for i in range(6)])
            if cate_id == 0 or cate_id == 11:
                continue
            srcimg_cate.append(cate_id)
            existed_gt.append((gt_left, gt_top, gt_width, gt_height))
        

        h, w, _ = tuple(src_img.shape)
        patches, patches_cate = patch_gen(15)
        all_cate = srcimg_cate + patches_cate 

        for patch in patches:
            patch_h, patch_w, _ = tuple(patch.shape)
            left = random.randint(0, w-patch_w)
            top = random.randint(0, h-patch_h)

            while overlapped_with_gt(left, top, patch_w, patch_h, existed_gt):
                left = random.randint(0, w-patch_w)
                top = random.randint(0, h-patch_h)

            src_img[top:top+patch_h, left:left+patch_w] = patch

            existed_gt.append( (left, top, patch_w, patch_h) )  #Newly added patch becomes part of existed_gt
        
        with open(hc_anno_path + '/' + 'hc_' + file_name.split('.')[0] + '.txt', "w") as hc_anno_f:
            for idx in range(len(existed_gt)):
                hc_anno_f.write(str(existed_gt[idx][0])+','+str(existed_gt[idx][1])+','+str(existed_gt[idx][2])+
                ','+str(existed_gt[idx][3])+',' + '_' + ',' + str(all_cate[idx])+'\n')
        hc_anno_f.close()
        existed_gt.clear()
        patches.clear()
        cv.imwrite('hard_chips' + '/' +'hc_'+ file_name, src_img)
        anno_file.close()


awning_tri = 3245
bicycle = 10469
bus = 5921
car = 114264
motor = 29622
pedestrian = 79205
people = 26998
tricycle = 4811
truck = 12858
van = 24937
nums = [pedestrian, people, bicycle, car, van, truck, tricycle, awning_tri, bus, motor]
total = sum(nums)

sample_ratio = [int((total-nums[i])/nums[i]) for i in range(10)]
sample_ratio_randn = [sum(sample_ratio[:i+1]) for i in range(10)]
#print(sample_ratio_randn)

def random_cate(sample_ratio_randn):
    randn = random.randint(0,293)
    for i in range(10):
        if randn - sample_ratio_randn[i] < 0:
            return category[i+1], i+1
    
    return 'pedestrian', 1


def patch_gen(n_patches):
    patches_num = n_patches
    patches = []
    patches_cate = []
    for n in range(patches_num):
        cate, cate_id = random_cate(sample_ratio_randn)
        patches_name = os.listdir(cate)
        img_randn = random.randint(1, len(patches_name))
        patch = cv.imread(cate + '/' + patches_name[img_randn-1])
        patches.append(patch)
        patches_cate.append(cate_id)
        patches_name.clear()

    return patches, patches_cate

patch_aug()
