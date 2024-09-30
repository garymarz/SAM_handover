import torch 
import cv2
import time
import os
import logging
import sys
import json
import datetime
import argparse
import shutil
import traceback

import numpy as np
import matplotlib.pyplot as plt

import time
from glob import glob
from segment_anything import sam_model_registry, SamPredictor,  SamAutomaticMaskGenerator
import threading

global args_output
# np.set_printoptions(threshold=np.inf)
# d:\sam\sam.exe d:\Models\CF5\123\*.pth d:\Models\CF5\123\input\ d:\Models\CF5\123\output 0
# pip install https://github.com/pyinstaller/pyinstaller/tarball/develop
# log -> log 寫在 model資料夾下
# input a.txt -> box, x1,y1,....xn,yn
# text l v h 模型 memery 占用大小

# Input
# txt \\10.91.45.57\DLADJloader$\INK\Input\image\CF5_INRE05_20230420101858_REPR.png,279,331,165,234,168,178  x1, y1, x2, y2 中心點記得算
# Out 
# \\10.91.45.57\DLADJloader$\INK\Input\image\CF5_INRE05_20230420101858_REPR.png,100%,365,404,304,340

# 原 IMAGE: CF5_INRE05_20230420101858_REPR.png
# MASK IMAGE: CF5_INRE05_20230420101858_REPR_contour.png

def model_predictor(model_type, checkpoint, device, img_path='test.png'):
    '''Define segmentation model. "Predictor" for single mask.  "Mask_generator" for anything.'''
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device=device)
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=16, points_per_batch=16)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image) # initial sam model
    return predictor, mask_generator

def show_mask(masks, image, input_point, box ,output_dir,random_color=True):
    '''Plot masks on the image'''
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i, mask in enumerate(masks):
            if random_color:
                colors = np.random.random(3)*250
                color = np.array([int(k) for k in colors],dtype=np.uint8) 
            else:
                color = np.array([200, 0, 0],dtype=np.uint8) 
            
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1).astype(np.uint8) * color.reshape(1, 1, -1)
            
            image = cv2.addWeighted(image, 1, mask_image, 0.5, 0)
            if box is not None:
                if box[i].any():
                    box0 = (int(box[i][0]), int(box[i][1]))
                    box1 = (int(box[i][2]), int(box[i][3]))
                    image = cv2.rectangle(image, box0, box1,(100, 50, 0), 2)
            if input_point is not None:
                if input_point[i].any():
                    point = (int(input_point[i][0]),int(input_point[i][1]))
                    image = cv2.circle(image, point, 3, (100, 50, 0),-1)
                    
        cv2.imwrite(output_dir, image)
    except Exception as err:
        print(err)
    
    
def show_defect(masks, input_box, output_dir):
    '''Write masks and merge masks. '''
    dir_path = output_dir
    url = []
    for i, j in enumerate(masks):
        # print(input_box[i])
        # name = "{}_{}_{}_{}.png".format(int(input_box[i][0]),int(input_box[i][2]),int(input_box[i][1]),int(input_box[i][3])) # 以box為名稱
        # http_path = "http://{}/".format(path)
        # \\\\169.254.82.11\\sam_api\\sam_output\\{shop}_{formattedDateTime}.json
        name0 = "{}_{}_{}_{}.png".format(int(input_box[i][0]),int(input_box[i][2]),int(input_box[i][1]),int(input_box[i][3]))
        name1 = "/Uploads/{}_{}_{}_{}.png".format(int(input_box[i][0]),int(input_box[i][2]),int(input_box[i][1]),int(input_box[i][3]))
        output_dir = os.path.join(dir_path,name0)
        
        #name1 = '/'+output_dir.split('\\')[-2]+'/'+output_dir.split('\\')[-1]
        colors = np.array([255,255,255])
        color = np.array([int(k) for k in colors],dtype=np.uint8)  
        mask = j[0]
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        # if i==0:
        #     masks = mask_image
        # else:
        #     masks = cv2.addWeighted(masks, 1, mask_image, 0.7, 0)
        if np.sum(mask_image)>0:
            cv2.imwrite(output_dir, mask_image)
            url.append(name1)
        # cv2.imwrite(output_dir0, masks)
    return url
        
def get_args(file= "config/config.json"):
    with open(file, 'r') as j:
        contents = json.loads(j.read())
    j.close()
    return contents
    
def txt_readfile(file= 'input/A.txt'):
    f = open(file, 'r')
    data = f.read()
    f.close()
    os.remove(file)
    return data

def write_output(img_dir, mask, persent, file):
    '''Save SAM output as txt file.'''
    if mask is None:
        txt = "{},seg1,0,0,0,0,0,".format(img_dir)
        return txt
    else:
        txt = "{},".format(img_dir)
        for t in range(len(mask)):
            points = np.where(mask[t][0]==True)
            try:
                x1,y1,x2,y2 = min(points[1]),min(points[0]),max(points[1]),max(points[0])
                txt = txt+"seg1,{:.1f}%,{},{},{},{},".format(persent[t][0]*100, x1,x2,y1,y2)
            except:
                txt = txt+"seg1,0,0,0,0,0,"
        return txt

def check_file(Dir:os.path):
    print(datetime.datetime.now(),'[Check] {}'.format(Dir))
    if not os.path.isdir(Dir):
        try:
            print(datetime.datetime.now(),'[Create] {}'.format(Dir))
            os.makedirs(Dir)
        except OSError as e:
            logger.error(e)
            
            
            
def check_checkpoint(Dir,checkpoint):
    if os.path.isfile(checkpoint):
        return checkpoint
    else:
        return os.path.join(Dir,checkpoint)
    
def check_tmp(Dir,checkpoint):
    if os.path.isdir(os.path.join(Dir,checkpoint)):
        return os.path.join(Dir,checkpoint)
    else:
        os.makedirs(os.path.join(Dir,checkpoint))
        return os.path.join(Dir,checkpoint)
    
def remove_file(Dir:os.path):
    '''Remove Dir/*'''
    filelist = glob(os.path.join(Dir, "*"))
    for f in filelist:
        os.remove(f)
        
def get_input(txt:str):
    '''Convert txt file to input_boxs、input_points and input_label'''
    if len(txt.split(',')) < 4:
        mode = 'anything'
        input_box, input_point, input_label = None, None, None
    else:
        if len(txt.split(',')) < 10:
            input_ = np.array([int(i) for i in txt.split(',')[1:7]])
            input_box = [input_[[0,2,1,3]]] # x1, x2, y1, y2 -> x1, y1, x2, y2
            input_point = np.array([[input_[4], input_[5]]])
            if txt.split(',')[8]=='':
                    input_label = [0]
            else:
                input_label = txt.split(',')[8]
            mode = 'boxes'
        else:
            input_boxs = np.array([])
            input_points = np.array([])
            input_labels = np.array([]) 
            for i in range(len(txt.split(','))//7):
                input_ = np.array([int(i) for i in txt.split(',')[7*i+1:7*i+7]])
                input_label = txt.split(',')[7*i+7]
                input_box = input_[[0,2,1,3]]
                input_point = np.array([[input_[4], input_[5]]])
                input_boxs = np.append(input_boxs, input_box)
                input_points = np.append(input_points, input_point)
                if input_label=='':
                    input_label = 0
                input_labels = np.append(input_labels, input_label)

            
            input_points = input_points.reshape(-1,2)
            input_boxs = input_boxs.reshape(-1,4)
            mode = 'boxes'
            return mode, input_boxs, input_points, input_labels
            
    return mode, input_box, input_point, input_label

def sam_predict(mode:str ,image:object , input_box:object, input_point:object, input_label:object, predictor:object, mask_generator:object):
    ''' SAM predictor:\n
        Only single predict use predictor.predict
    '''

    if mode=='box':
        predictor.set_image(image)
        box_ = input_box[0]
        input_p = input_point[0]
        if sum(input_p==0)==2:input_p = None
        if sum(box_==0)==4:box_ = None
        input_label = [0]

        masks, scores, logits = predictor.predict(
                                            point_coords=input_p,
                                            point_labels=input_label,
                                            box= box_,
                                            multimask_output=False,)
        masks = masks.reshape(1, 1, masks.shape[-2],masks.shape[-1])
        scores = np.array(scores).reshape(-1,1)
        logits = np.array(logits).reshape(-1,1)

    if mode == 'boxes':
        print(datetime.datetime.now(),'Multiple boxes mode')
        predictor.set_image(image)
        masks, scores, logits = np.array([]), np.array([]), np.array([])
        for i in range(len(input_box)):
            input_p = input_point[i].reshape(-1,2)
            #print(input_p)
            box_ = input_box[i]
            if sum(input_p[0]==0)==2:input_p = None
            if sum(box_==0)==4:box_ = None
            input_label = [0]
            mask, score, logit = predictor.predict(
                                            point_coords=input_p,
                                            point_labels=input_label,
                                            box=box_,
                                            multimask_output=False,)
            mask = mask.reshape(1, mask.shape[-2],mask.shape[-1])
            masks = np.append(masks, mask)
            scores = np.append(scores, score)
            logits = np.append(logits, logit)
        masks = masks.reshape(-1, 1, mask.shape[-2],mask.shape[-1])
        scores = np.array(scores).reshape(-1,1)
        logits = np.array(logits).reshape(-1,1)

    if mode == 'anything':
        print(datetime.datetime.now(),'Anything mode')
        mask = mask_generator.generate(image)
        masks, scores, logits = [], [], []
        for i in mask:
            masks.append(i['segmentation'])
            scores.append(i['stability_score'])
            logits.append(i['predicted_iou'])

        masks = np.array(masks)
        w, h = masks.shape[-2], masks.shape[-1]
        masks = masks.reshape(-1,1,w,h)
        scores = np.array(scores).reshape(-1,1)
        logits = np.array(logits).reshape(-1,1)
        
    return masks, scores, logits
        

def contourtojson(masks:object , input_box:object, input_point:object, imgpath:str, Url:str):
    '''
    Save contour to json format.\n
    {
    "imageName": "xxx01.jpg",\n
    "resultUrl": ["http://xxxx.xxx.xxx.xxx/11_33_22_44.jpg"]\n
    "labels" : {
	            "11_33_22_44" : [[1,1],[5,6],[18,52],[20,60],[6,10]],\n 
                "11_333_222_444" : [[1,1],[5,6],[18,52],[20,60],[6,10]]
               }}
    '''
    format = {"imageName":imgpath,"resultUrl": [],"labels":{} }
    for i in range(len(masks)):
        contours, _ = cv2.findContours(np.array(masks[i][0], np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #img = cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
        #cv2.imwrite('D:\\yangu\\dataset\\sam_api\\sam_output\\123.jpg',img)
        contour_name = ''
        if sum(input_box[i]==0)==4:
            contour_name = ''
        else:
            x1,x2,y1,y2=input_box[i][0],input_box[i][2],input_box[i][1],input_box[i][3]
            contour_name = '{}_{}_{}_{}'.format(int(x1),int(x2),int(y1),int(y2))
        if sum(input_point[i]==0)!=2:
            x1,x2,y1,y2=input_box[i][0],input_box[i][2],input_box[i][1],input_box[i][3]
            x,y = input_point[i][0],input_point[i][1]
            contour_name =  '{}_{}_{}_{}_{}_{}'.format(int(x1),int(x2),int(x),int(y1),int(y2),int(y))

        format["labels"][contour_name]=[i.reshape(-1,2).tolist() for i in contours if len(i)>5]
        
    if Url == "output_image\n":
        urls = show_defect(masks, input_box, args.output)
        #print(urls)
        #test = 'http://169.254.82.11/sam_api/sam_output/'
        
        #name = test+urls.split('\\')[-1]
        #name = [args_output+'\\'+i.split('\\')[-1] for i in urls]
        format["resultUrl"] = urls
        
    return format

def cleanup_folder(folder_path:str, max_files=300, files_to_keep=10):
    '''Clean folder if 'max_files' > 100
    '''
    try:
        # 檢查資料夾是否存在
        if not os.path.exists(folder_path):
            print("資料夾不存在")
            return
        
        # 獲取資料夾內所有檔案的清單
        all_files = glob(os.path.join(folder_path, "*"))
        
        # 如果檔案數量超過最大限制，則刪除最早的一些檔案
        if len(all_files) > max_files:
            files_to_delete = sorted(all_files)[:len(all_files) - files_to_keep]
            for file_to_delete in files_to_delete:
                os.remove(file_to_delete)
                print(f"刪除檔案: {file_to_delete}")
    except Exception as e:
        print("發生錯誤:", str(e))


# load SAM model
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type = str, default = 'D:\\SAM\\input', help = "path to the input/a.txt")
    parser.add_argument("--output", type = str, default = 'D:\\SAM\\output',  help = "path to the output")
    parser.add_argument("--output_url", type = str, default = 'http://169.254.82.11/sam_api/sam_output',  help = "path to the output")
    parser.add_argument("--device", type = str, default ='2', help = "cuda number")
    parser.add_argument("--checkpoint", type = str, default = 'vit_l.pth', help = "path to the checkpoint")
    parser.add_argument("--cuda_capacity", type = float, default = 0.7, help = "the percentage of cuda_capacity")
    args = parser.parse_args()
    print(args)
    device = 'cuda:{}'.format(args.device)
    input_dir = args.input
    output_dir = args.output
    #check_file(os.path.join(output_dir,'ct_image'))
    checkpoint = args.checkpoint
    model_type = checkpoint.split('\\')[-1].split('.')[0]
    #torch.cuda.set_per_process_memory_fraction(float(args.cuda_capacity), int(device[-1])) # only can use 70% GPU memery
    # urlss = {  'FAB1':r'http://p1cimweb.cminl.oa/ApiGateway/SamApi/Uploads/',
    #            'FAB2':r'http://p2cimweb.cminl.oa/ApiGateway/SamApi/Uploads/',
    #            'FAB3':r'http://p3cimweb.cminl.oa/ApiGateway/SamApi/Uploads/',
    #            'FAB4':r'http://p4cimweb.cminl.oa/ApiGateway/SamApi/Uploads/',
    #            'FAB5':r'http://p5cimweb.cminl.oa/ApiGateway/SamApi/Uploads/',
    #            'FAB6':r'http://p6cimweb.cminl.oa/ApiGateway/SamApi/Uploads/',
    #            'FAB7':r'http://p7cimweb.cminl.oa/ApiGateway/SamApi/Uploads/',
    #            'FAB8':r'http://p8cimweb.cminl.oa/ApiGateway/SamApi/Uploads/',
    #            'T6'  :r'http://plcimweb.cminl.oa/ApiGateway/SamApi/Uploads/',
    #            'OA'  :r'http://tncimweb.cminl.oa/ApiGateway/SamApi/Uploads/'
    #         }
               


    f_name = str(datetime.date.today())
    log_Path = os.path.dirname(input_dir)
    log_ = check_tmp(log_Path,'log')
    tmp = check_tmp(log_Path,'tmp')
    contour = check_tmp(log_Path,'contour')
    check_file(log_)
    logging.basicConfig(level = logging.DEBUG,
                    format = '[%(levelname)s] %(asctime)s %(message)s',
                    datefmt ='%Y-%m-%d %H:%M:%S',
                    filename = os.path.join(log_,str(os.getpid())+'.log'),
                    filemode = 'a')
    logger = logging.getLogger(__name__)
    # model initial
    print(datetime.datetime.now(), "SAM Model Initial")
    checkpoint_path = check_checkpoint(log_Path, checkpoint)
    img_path = os.path.join(os.path.dirname(checkpoint), 'test.png')
    predictor, mask_generator = model_predictor(model_type, checkpoint_path, device, img_path)
    
    remove_file(tmp) # 流程失敗卡在TMP該如何處理
    while True:
        time.sleep(0.1)
        p = glob(input_dir+'/*.txt')
        
        try:
            print(datetime.datetime.now(), 'Waiting...')
            if len(p)>0:
                for path0 in p:
                    try:
                        # s = time.time()
                        #fab = path0.split('\\')[-1].split('_')[0] # 廠區
                        args_output = ''
                        txt_path = shutil.move(os.path.join(path0),tmp)
                        print(datetime.datetime.now(),"Get process file {}".format(path0))
                        logger.info("Find process file {}".format(txt_path))
                        json_file = []
                        if os.path.isfile(txt_path):
                            with open(txt_path, 'r', encoding='UTF-8') as f:
                                lines = f.readlines()
                                write_txt = ''
                                
                                for txt in lines:
                                    img_name = txt.split(',')[0].split('.')[0]
                                    urls = txt.split(',')[-1]
                                    
                                    out_file = os.path.join(output_dir, img_name)
                                    file_name, extension = os.path.splitext(os.path.basename(txt.split(',')[0]))
                                    image_name = txt.split(',')[0] # 注意檔案位置
                                    mode, input_box, input_point, input_label = get_input(txt)
                                    
                                    
                                    image = cv2.imread(input_dir+'/'+image_name)
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    masks, scores, logits = sam_predict(mode, image, input_box, input_point, input_label, predictor, mask_generator)
                                    
                                    print(datetime.datetime.now(), "Ploting...")
                                    contour_s = contourtojson(masks, input_box, input_point, image_name, urls)
                                    json_file.append(contour_s)
                                    
                                    os.remove(input_dir+'/'+image_name) # 做完刪掉圖片
                                    # show_defect(masks, input_box, args.output) # 不一定都要這個
                                    
                            tmp_name = output_dir+'/'+path0.split('.')[0].split('\\')[-1]+".tmp"
                              
                            with open(tmp_name , "w") as f:
                                json.dump(json_file, f)
                            f.close()
                            # s3 = time.time()
                            # print('寫成json並存圖',s3-s2)
                            print(datetime.datetime.now(), "Finish !")
                            if os.path.exists(tmp_name):
                                os.rename(tmp_name , output_dir+'/'+path0.split('.')[0].split('\\')[-1]+".json")
                            # 刪json
                            cleanup_folder(output_dir)    
                            
                            
                    except Exception as err:
                        logger.error(err)
                        print(err)
                        
                        
        except Exception as err:
            logger.error(err)
            print(err)
            
    