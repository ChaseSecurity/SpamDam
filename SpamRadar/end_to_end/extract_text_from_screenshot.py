from functools import reduce
import re
import cv2 as cv
import pytesseract
import copy
import matplotlib.pyplot as plt
import pandas as pd

# match phonenumber
pattern = re.compile('\+[0-9]*')

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)    
    for i in r:
        input_txt = input_txt.replace(i,'')    
    return input_txt 

def reverse_color(img):
    img = 255 - img.copy()
    return img

def deleteDuplicate(list_):
    
    func = lambda x, y: x if y in x else x + [y]
    list_ = reduce(func, [[], ] + list_)
    list_ = pd.DataFrame(list_)
    list_ = list_.drop_duplicates(subset=[1,2,3,4]).values.tolist()
    
    return list_

def get_boxes(boxes):
    
    text = boxes[0]
    left = int(boxes[1])
    bottom = int(boxes[2])
    right = int(boxes[3])
    top = int(boxes[4])
    conf = boxes[5]
    
    return text, left, bottom, right, top, conf

def get_conf_box(data, i, h, w):
    tmp = []
    if data['conf'][i] >= 65 or data['conf'][i] == 0 :
        text = data['text'][i]
    else:
        text = '\xbf' # X
    tmp.append(text)
    tmp.append(data['left'][i])
    tmp.append(h-data['top'][i]-data['height'][i])
    tmp.append(data['left'][i]+data['width'][i])
    tmp.append(h - data['top'][i])
    tmp.append(data['conf'][i])
    return tmp

def push_box(boxes, text, left, bottom, right, top, conf):
    tmp = []
    tmp.append(text)
    tmp.append(left)
    tmp.append(bottom)
    tmp.append(right)
    tmp.append(top)
    tmp.append(conf)
    boxes.append(tmp)

def merge_box(front, rare, lang):
    
    text_front, left_front, bottom_front, right_front, top_front, conf_front = get_boxes(front)
    text_rare, left_rare, bottom_rare, right_rare, top_rare, conf_rare = get_boxes(rare)
    
    left_max = min(left_front, left_rare)
    right_max = max(right_rare, right_front)
    top_max = max(top_front, top_rare)
    bottom_max = min(bottom_front, bottom_rare)
    if lang != 'chi_sim':
        text_max = text_front + ' '+text_rare
    else:
        text_max = text_front + ' '+text_rare
    conf_min = min(conf_front, conf_rare)
    if conf_min<65:
        conf_min = max(conf_front, conf_rare)
    
    return text_max, left_max, bottom_max, right_max, top_max, conf_min

def compute_area(box):
    width = box[3] - box[1]
    height = box[4] - box[2]
    return width * height


def func(img):
    #把图片转换为灰度图
    gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY);
    #获取灰度图矩阵的行数和列数
    r,c = gray_img.shape[:2];
    dark_sum=0; #偏暗的像素 初始化为0个
    dark_prop=0;  #偏暗像素所占比例初始化为0
    piexs_sum=r*c; #整个弧度图的像素个数为r*c

    #遍历灰度图的所有像素
    for row in gray_img:
        for colum in row:
            if colum<40: #人为设置的超参数,表示0~39的灰度值为暗
                dark_sum+=1;
    dark_prop=dark_sum/(piexs_sum); 
#     print("dark_sum:"+str(dark_sum));
#     print("piexs_sum:"+str(piexs_sum));
#     print("dark_prop=dark_sum/piexs_sum:"+str(dark_prop));
    if dark_prop >=0.75:  #人为设置的超参数:表示若偏暗像素所占比例超过0.78,则这张图被认为整体环境黑暗的图片
        return 'dark'
    else:
        return 'bright'

def extract_text_from_screenshot(filename, lang='chi_sim'):
    # read the image and get the dimensions
    img = cv.imread(filename)
    if func(img) == 'dark':
        img = reverse_color(img)
    h, w, _ = img.shape # assumes color image
    
    # run tesseract, returning the bounding boxes
    data = pytesseract.image_to_data(img, lang = '{}+eng'.format(lang), output_type=pytesseract.Output.DICT, config='--oem 1')

    box_data = []
    for i in range(len(data['conf'])):
        if data['conf'][i] == -1:
            continue
        tmp = get_conf_box(data,i, h, w)
        box_data.append(tmp)

    # draw the bounding boxes on the image
    horizontal_merge = []
    if lang == 'chi_sim' or lang== 'kor' or lang == 'jpn':
        horizontal_threshold = 43
    else:
        horizontal_threshold = 30
    # merge boxes horizonal
    for i in range(1,len(box_data)):

        front = box_data[i-1]
        rare = box_data[i]

        text_front, left_front, bottom_front, right_front, top_front, conf_front = get_boxes(front)
        text_rare, left_rare, bottom_rare, right_rare, top_rare, conf_rare = get_boxes(rare)

        if i ==1 :
            push_box(horizontal_merge, text_front, left_front, bottom_front, right_front, top_front, conf_front)

        get_data = horizontal_merge.pop()
        text_front, left_front, bottom_front, right_front, top_front, conf_front = get_boxes(get_data)

        if (bottom_rare > (bottom_front - horizontal_threshold)) and top_rare > bottom_front and top_rare < (top_front + horizontal_threshold) and (right_front + horizontal_threshold) > left_rare and right_rare > (left_front - horizontal_threshold ):
            text_max, left_max, bottom_max, right_max, top_max, conf_min = merge_box(get_data, rare, lang)
            push_box(horizontal_merge, text_max, left_max, bottom_max, right_max, top_max, conf_min)
        else:
            push_box(horizontal_merge, text_front, left_front, bottom_front, right_front, top_front, conf_front)
            push_box(horizontal_merge, text_rare, left_rare, bottom_rare, right_rare, top_rare, conf_rare)

    # merge boxes vertical
    horizontal_copy = copy.deepcopy(horizontal_merge)
    results = []
    if lang == 'chi_sim'or lang== 'kor':
        if h > 2000: 
            height_threshold = 33
        else:
            height_threshold = 17
    elif lang == 'jpn':
        height_threshold = 19
    else:
        height_threshold = 32
    for i in range(0,len(horizontal_copy)):
        front = horizontal_copy[i]
        text_front, left_front, bottom_front, right_front, top_front, conf_front = get_boxes(front)
        text_max, left_max, bottom_max, right_max, top_max, conf_min = get_boxes(front)

        # in some cases the boxes which are overlaped are not adjacent
        for j in range(i+1, len(horizontal_copy)):
            if j == i+1:
                front_tmp = front
                
            rare = horizontal_copy[j]
            text_rare, left_rare, bottom_rare, right_rare, top_rare, conf_rare = get_boxes(rare)
            text_front, left_front, bottom_front, right_front, top_front, conf_front = get_boxes(front_tmp)
            
            if (top_rare >= (bottom_front - height_threshold)) and (right_front ) > left_rare and right_rare > (left_front - horizontal_threshold):
    
                text_max, left_max, bottom_max, right_max, top_max, conf_min = merge_box(front_tmp, rare, lang)
                tmp = []
                tmp.append(text_max)
                tmp.append(left_max)
                tmp.append(bottom_max)
                tmp.append(right_max)
                tmp.append(top_max)
                tmp.append(conf_min)
                
                horizontal_copy[j][1:-1] = tmp[1:-1]
                front_tmp = tmp
    
        push_box(results, text_max, left_max, bottom_max, right_max, top_max, conf_min)
    results = deleteDuplicate(results)
    if len(results) == 0:
        return
    sms_text = results[0][0]
    max_area = compute_area(results[0])
    max_text = len(sms_text)
    
    # results: 放结果框出的的大框
    # horizontal: 水平合并的框，下面换掉循环可以显示图片
    
    # 显示效果图
#     for n in results:
#         img = cv.rectangle(img, (int(n[1]), h - int(n[2])), (int(n[3]), h - int(n[4])), (0, 255, 0), 3)
#     plt.imshow(img)
#     plt.show()
    
    
#     for i in horizontal_merge:
#         print(i)
    for i in results:
#         if compute_area(i) > max_area:
#             max_area = compute_area(i)
#             sms_text = i[0]
#         else:
        if len(remove_pattern(i[0], pattern)) > max_text:
            sms_text = i[0]
            max_text = len(remove_pattern(i[0], pattern))
#        print(i[0])
#         print(len(remove_pattern(i[0], pattern)))
    return sms_text