#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import argparse
import base64
import json
import os
import collections
import time

import requests
import multiprocessing as mp
from functools import partial
import tqdm
import cv2
import numpy as np

"""
标注给时间和图片， 在已经截取的时间段+-2s内查找相似度最高的3个人，如果匹配成功，则将所有图片放在一个文件夹内，若没有匹配则只放标注的图片到文件夹内
文件夹已标注时间为文件名
"""


def _get_crop_time(file_name):
    """
    从文件夹名称中找到图片截取时间
    :param file_name:
    :return:
    """
    # print(file_name)
    result = 0
    try:
        time_str = file_name.split('_')[-1]
        result = int(time_str.split('-')[1])
    except Exception as e:
        pass
        # print(' file name {} is invalid'.format(file_name))

    return result


class TrackObject():
    def __init__(self, file_name):
        self.file_name = file_name
        self.cropped_time = _get_crop_time(file_name)

    def __str__(self):
        return str(self.cropped_time)+':'+self.file_name


class FacePic():
    def __init__(self, file_path):
        self.file_path = file_path
        try:
            pic_name = os.path.basename(file_path)
            self.quality = float(pic_name.split('_')[-1].split('G')[0])
        except Exception as e:
            self.quality = 0.0

    def __str__(self):
        return self.file_path+':'+str(self.quality)


def find_the_best_quality_pic(dir_path):
    pic_list = []
    for f_name in os.listdir(dir_path):
        f_type = f_name.split('.')[-1]
        if f_type == 'jpg':
            pic_list.append(FacePic(os.path.join(dir_path, f_name)))

    if len(pic_list) == 0:
        return None
    else:
        pic_list.sort(key=lambda x: x.quality)
        return pic_list[0]


def get_ordered_filepaths(parent_path):
    result_list = []
    result_dict = collections.OrderedDict()
    paths = os.listdir(path=parent_path)
    for pa in paths:
        obj = TrackObject(pa)
        if obj.cropped_time != 0:
            result_list.append(TrackObject(pa))
    result_list.sort(key=lambda x: x.cropped_time)
    for i in result_list:
        result_dict[i.cropped_time] = i.file_name

    return result_list, result_dict


def find_in_range_best_image(face_server_url, parent_path, origin_pic_path):
    """
    time + - 5内产生的图片, 每个track拿出质量最好的，然后和face-x进行对比， 返回top3
    图片标注规则
    1.进门人数（实际进门人数，如供餐工作人员频繁进出记为1人）
    2.进门人次（同一人重复进门记多次）
    3.进门时间
    4.图片大小 256*341 图片格式是 .jpg. 图片命名方式： 随意不重复字符串_人脸被截图的时间.jpg 例子：p01_123818.jpg
    :param face_server_url: faceserver 地址
    :param parent_path:  前置摄像头内的图片文件路径
    :param origin_pic_path: 标注的图片路径
    :return:
    """
    origin_time = str(os.path.basename(origin_pic_path).split('_')[-1]).split('.')[-2]
    file_path_list, file_path_dict = get_ordered_filepaths(parent_path)
    files = [x for x in file_path_list if int(origin_time) + 5 > x.cropped_time > int(origin_time) - 5]
    pics = []
    not_paired_pic = []
    # print('在氛围的文件有{}个'.format(len(files)))
    for f in files:
        path = os.path.join(parent_path, f.file_name)
        best_quality_pic = find_the_best_quality_pic(path)
        if best_quality_pic is not None:
            # print(type(origin_pic_path), origin_pic_path)
            # print(type(best_quality_pic), best_quality_pic)
            res = compare_1v1(face_server_url, origin_pic_path, best_quality_pic)
            if res['error_no'] == 0 and res['data']['score'] > 70:
                pics.append((best_quality_pic, res['data']['score']))
            else:
                not_paired_pic.append((best_quality_pic, 0))
    # print(origin_pic_path, len(pics), len(not_paired_pic))
    return origin_pic_path, pics, not_paired_pic


def compare_1v1(url, image1_path, image2_obj):
    url = 'http://'+url+'/face/v1/compare-two-image'
    headers = {
        'Content-Type': "application/json",
        'cache-control': "no-cache"
    }
    payload = {
        'check': False,
        'image1': tobase64(image1_path),
        'image2': tobase64(image2_obj.file_path)
    }
    response = requests.request("POST", url, data=json.dumps(payload), headers=headers).json()
    return response


def tobase64(image_path):
    with open(image_path, 'rb') as f:
        image_64 = base64.b64encode(f.read()).decode('utf-8').replace("'", "")
        return image_64


def imgs_merge(first_image, merge_image_list, output, file_name):
    left = cv2.imread(first_image)
    res_image = left
    # print('imgs_merge' + str(len(merge_image_list)))
    for image in merge_image_list:
        right = cv2.imread(image)
    # left = cv2.resize(left, (int(left.shape[1] * 400 / left.shape[0]), 400), interpolation=cv2.INTER_CUBIC)
    # right = cv2.resize(right, (int(right.shape[1] * 400 / right.shape[0]), 400), interpolation=cv2.INTER_CUBIC)

        res_image = np.concatenate((res_image, right), axis=1)
    # print(output, file_name)
    img_name = os.path.join(output, file_name)
    cv2.imwrite(img_name, res_image)


def compare_marked_with_output(marked_pic_path, did_pic_path, out_put_path, face_server, ncpu=5):
    randomstr = str(int(time.time()))
    marked_pics_name = os.listdir(marked_pic_path)
    print("Start compare marked pics: [{}]".format(len(marked_pics_name)))

    # m_pic = marked_pics_name[1]
    # res = find_in_range_best_image(dic_pic_path, os.path.join(marked_pic_path, m_pic))
    # print(res)
    marked_pic_paths = [os.path.join(marked_pic_path, x) for x in marked_pics_name]
    func = partial(find_in_range_best_image, face_server, did_pic_path)
    with mp.Pool(ncpu) as p:
        results = list(tqdm.tqdm(p.imap(func, marked_pic_paths), total=len(marked_pic_paths)))

    count1 = 0
    count2 = 0
    out_put_path_random = os.path.join(out_put_path, randomstr)
    for res in results:
        if res is not None and res[1] is not None:
            if len(res[1]) == 0:
                count1 += 1
                print(res[0], "No pic paired")

            if len(res[1]) == 0 and len(res[2]) != 0:
                not_paired_save_path = out_put_path_random+'_not_paired'
                if not os.path.exists(not_paired_save_path):
                    os.makedirs(not_paired_save_path)
                merge_file_paths = []
                for index, value in enumerate(res[2]):
                    merge_file_paths.append(value[0].file_path)
                imgs_merge(res[0], merge_file_paths, not_paired_save_path, os.path.basename(res[0]))
            elif len(res[1]) != 0:
                count2 += 1
                if not os.path.exists(out_put_path_random):
                    os.makedirs(out_put_path_random)
                # print('out' + out_put_path_random)
                merge_file_paths = []
                for index, value in enumerate(res[1]):
                    merge_file_paths.append(value[0].file_path)
                print(res[0], "paired with {} pictures".format(len(merge_file_paths)))
                imgs_merge(res[0], merge_file_paths, out_put_path_random, os.path.basename(res[0]))
    print('has {} paired, {} unpaired'.format(count2, count1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compared marked pic with did generated pics')
    parser.add_argument('--marked', help='marked_pic_path', default='/Users/guorong/Documents/office/marked_pic_path')
    parser.add_argument('--did', help='did_pic_path', default='/Users/guorong/Documents/office/pic_from_did/data/20190312-103338-848')
    parser.add_argument('--output', help='out_put_path', default='/Users/guorong/Documents/office/paired')
    parser.add_argument('--faceserver', help='face server address', default='172.16.10.1:8096')
    args = parser.parse_args()
    marked_pic_path = args.marked
    did_pic_path = args.did
    out_put_path = args.output
    face_server = args.faceserver
    compare_marked_with_output(marked_pic_path, did_pic_path, out_put_path, face_server)

