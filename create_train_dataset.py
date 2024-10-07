from glob import glob
import shutil
import random
import numpy as np
from skimage import transform, io, segmentation
import json
import shutil
random.seed(123)
# imgpath ="D:\\yangu\\dataset\\sam_png_images\\"
path0 = glob('D:\\yangu\\dataset\\background_mask_all\\*') # normal background
path1 = glob('D:\\yangu\\dataset\\background_mask_with_defect\\*') # background with defect
path2 = glob('D:\\yangu\\dataset\\background_mask_with_defect_full\\*') # full background with defect
path0 = random.sample(path0,3200)
path1 = random.sample(path1,3200)
path2 = random.sample(path2,3200)

path = path0[:3000] + path1[:3000] + path2[:3000]
patht = path0[3000:] + path1[3000:] + path2[3000:]

ds = {'train':path,'val':patht}

for k in ds.keys():
    for i in ds[k]:
        try:
            name = i.split('\\')[-1].split('.')[0].split('_')
            image_name = '{}_{}_{}_{}_{}.png'.format( *name[:5])
            gt_name = i.split('\\')[-1]
            j_name = 'D:/yangu/dataset/files_json/{}/{}.json'.format(k, i.split('\\')[-1].split('.')[0])
            gt_data = io.imread(i)
            if len(gt_data.shape) == 3:
                gt_data = gt_data[:, :, 0]
            height, width = gt_data.shape
            shutil.copyfile(i, 'D:\\yangu\\dataset\\files_json\\masks\\'+gt_name )
            # if np.sum(gt_data>0) >0:
            #     points = np.where(gt_data>0)
            #     points = [[x,y] for x,y in zip( *points)]
            #     point = random.sample(points,1)[0]
            
            y_indices, x_indices = np.where(gt_data > 0)
            points = [[x,y] for y,x in zip( *np.where(gt_data > 0))]
            points = random.sample(points,1)[0]

            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            x = np.random.randint(1, 10)
            y = np.random.randint(1, 10)
            x_min = max(0, x_min - x)
            x_max = min(width, x_max + x)
            y_min = max(0, y_min - y)
            y_max = min(height, y_max + y)
            box = [x_min, y_min, x_max, y_max]
            gt_data = gt_data.tolist()
            points = np.uint32(points).tolist()
            
            box = np.uint32(box).tolist()
            gtname = i.split('\\')[-1]
            d = { "image_name": image_name,
                  "gt_data": gt_name,
                  "points": points,
                  "labels": name[-2],
                  "boxes": box,
                  "height": height,
                  "weight": width }

            with open(j_name , "w") as f:
                json.dump(d, f)
            f.close()
        
        except Exception as err:
            print(err)  
