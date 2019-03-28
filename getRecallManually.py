#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import argparse
import base64
import json
import os
import collections
import time
from datetime import datetime

import requests
import multiprocessing as mp
from functools import partial
import tqdm
import cv2
import numpy as np
import pytz


"""
标注给时间和图片， 在已经截取的时间段+-2s内查找相似度最高的3个人，如果匹配成功，则将所有图片放在一个文件夹内，若没有匹配则只放标注的图片到文件夹内
文件夹已标注时间为文件名
"""


def _get_crop_time(file_path):
    """
    从文件夹名称中找到图片截取时间
    :param file_name:
    :return:
    """
    # print(file_path)
    result = 0
    tz = pytz.timezone('Asia/Shanghai')
    try:
        file_list = os.listdir(file_path)
        for f in file_list:
            fe = f.split('.')[-1]
            # print(fe)
            if fe == 'log':
                with open(os.path.join(file_path, f), 'r') as log:
                    log_json = json.loads(log.read())
                    first_time = int(json.loads(log_json['request']['body'])['first_time']/1000)
                    # print(first_time)
                    # print(first_time)
                    result = int(datetime.fromtimestamp(first_time, pytz.timezone('Asia/Shanghai')).strftime('%H%M%S'))
                    # print('result' + result)
                break
    except Exception as e:
        pass
    # print(result)
    return result


def _get_track_num(file_path):
    # print(file_path)
    # str = 'PROJredstar_AREA10062_HOST1_CH26_CAM172.16.10.28_FP20190319-162808-001'
    try:
        chnum = int(os.path.basename(file_path).split('_')[3].replace('CH', ''))
        return chnum
    except Exception as e:
        return None


class TrackObject:
    def __init__(self, file_path):
        self.file_name = os.path.basename(file_path)
        self.cropped_time = _get_crop_time(file_path)
        self.track_num = _get_track_num(file_path)

    def __str__(self):
        return str(self.cropped_time)+':'+self.file_name


class FacePic:
    def __init__(self, file_path):
        self.file_path = file_path
        try:
            pic_name = os.path.basename(file_path)
            self.quality = float(pic_name.split('_')[-1].split('G')[0])
        except Exception as e:
            self.quality = 0.0

    def __str__(self):
        return self.file_path+':'+str(self.quality)


def _find_the_best_quality_pic(dir_path):
    pic_list = []
    for f_name in os.listdir(dir_path):
        f_type = f_name.split('.')[-1]
        if f_type == 'jpg' and 'FromPhoto' not in f_name:
            pic_list.append(FacePic(os.path.join(dir_path, f_name)))

    if len(pic_list) == 0:
        return None
    else:
        pic_list.sort(key=lambda x: x.quality, reverse=True)
        return pic_list[0]


def _get_ordered_filepaths(parent_path, track_num=None):
    """
    给定track number ,输出已cropped_time排序的列表
    :param parent_path:
    :param track_num: 找到含有这个track num 的视频截图  track_num 对应文件夹中文件名中CH后面的数字
    :return:
    """
    result_list = []
    result_dict = collections.OrderedDict()
    paths = os.listdir(path=parent_path)
    for pa in paths:
        obj = TrackObject(os.path.join(parent_path, pa))
        # print(obj.cropped_time)
        if track_num is not None and obj.cropped_time != 0 and obj.track_num == track_num:
            result_list.append(obj)
        if track_num is None and obj.cropped_time != 0:
            result_list.append(obj)
    result_list.sort(key=lambda x: x.cropped_time)
    for i in result_list:
        result_dict[i.cropped_time] = i.file_name
    # print(result_dict)
    return result_list, result_dict


def _compare_1v1(url, image1_path, image2_obj):
    """
    在faceserver中比对，看看是不是同一个人
    :param url:
    :param image1_path:
    :param image2_obj:
    :return:
    """
    url = 'http://'+url+'/face/v1/compare-two-image'
    headers = {
        'Content-Type': "application/json",
        'cache-control': "no-cache"
    }
    payload = {
        'check': False,
        'image1': _tobase64(image1_path),
        'image2': _tobase64(image2_obj.file_path)
    }
    response = requests.request("POST", url, data=json.dumps(payload), headers=headers).json()
    return response


def _tobase64(image_path):
    with open(image_path, 'rb') as f:
        image_64 = base64.b64encode(f.read()).decode('utf-8').replace("'", "")
        return image_64


def _imgs_merge(first_image, merge_image_list, output, file_name):
    left = cv2.imread(first_image)
    res_image = cv2.resize(left, (int(left.shape[1] * 400 / left.shape[0]), 400), interpolation=cv2.INTER_CUBIC)
    # print('imgs_merge' + str(len(merge_image_list)))
    for image in merge_image_list:
        right = cv2.imread(image)
        right = cv2.resize(right, (int(right.shape[1] * 400 / right.shape[0]), 400), interpolation=cv2.INTER_CUBIC)
        res_image = np.concatenate((res_image, right), axis=1)

    # print(output, file_name)
    img_name = os.path.join(output, file_name)
    cv2.imwrite(img_name, res_image)


def _find_in_range_best_image(face_server_url, parent_path, track_num, fluct, origin_pic_path):
    """
    fluct时间内产生的图片, 每个track拿出质量最好的，然后和face-x进行对比， 返回top3
    :param face_server_url: faceserver 地址
    :param parent_path:  前置摄像头内的图片文件路径
    :param origin_pic_path: 标注的图片路径
    :param track_num: 测试第几路的recall
    :return:
    """
    # print(origin_pic_path)
    origin_time = str(os.path.basename(origin_pic_path).split('_')[-1]).split('.')[-2]
    file_path_list, file_path_dict = _get_ordered_filepaths(parent_path, track_num)
    files = [x for x in file_path_list if (int(origin_time) + fluct) > x.cropped_time > (int(origin_time) - fluct)]
    pics = []
    not_paired_pic = []
    # print('在氛围的文件有{}个'.format(len(files)))
    for f in files:
        path = os.path.join(parent_path, f.file_name)
        best_quality_pic = _find_the_best_quality_pic(path)
        if best_quality_pic is not None:
            # print(type(origin_pic_path), origin_pic_path)
            # print(type(best_quality_pic), best_quality_pic)
            res = _compare_1v1(face_server_url, origin_pic_path, best_quality_pic)
            if res['error_no'] == 0 and res['data']['score'] > 70:
                pics.append((best_quality_pic, res['data']['score']))
            elif res['error_no'] == 0 and res['data']['score'] <= 70:
                not_paired_pic.append((best_quality_pic, res['data']['score']))
            else:
                not_paired_pic.append((best_quality_pic, 0))
    # print(origin_pic_path, len(pics), len(not_paired_pic))
    return origin_pic_path, pics, not_paired_pic


def compare_marked_with_output(marked_pic_path, did_pic_path, out_put_path, face_server, track_num=None, ncpu=5, fluct=10):
    """
    图片标注规则
    1.进门人数（实际进门人数，如供餐工作人员频繁进出记为1人）
    2.进门人次（同一人重复进门记多次）
    3.进门时间
    4.图片大小 256*341 图片格式是 .jpg. 图片命名方式： 随意不重复字符串_人脸被截图的时间.jpg 例子：p01_123818.jpg

    :param marked_pic_path: 已经标注的图片路径， 要求图片的文件名: 随机唯一值_OCR时间命名  例子： p1_081123.jpg
    :param did_pic_path: VIP SDK 截取到的图片集合
    :param out_put_path: 结果输出目录
    :param face_server: face server HTTP 服务地址
    :param track_num: 第几路视频，从0开始计算
    :param ncpu: 并行计算使用的CPU数量
    :param fluct: 根据标注人脸的OCR时间，上下浮动几秒到视频的截图中查找人脸
    :return: 输出匹配日志和比对人脸
    """
    randomstr = str(int(time.time()))
    marked_pics_name = os.listdir(marked_pic_path)
    print("Start compare marked pics: [{}]".format(len(marked_pics_name)))

    marked_pic_paths = [os.path.join(marked_pic_path, x) for x in marked_pics_name]
    func = partial(_find_in_range_best_image, face_server, did_pic_path, track_num, fluct)
    with mp.Pool(ncpu) as p:
        results = list(tqdm.tqdm(p.imap(func, marked_pic_paths), total=len(marked_pic_paths)))

    # 统计结果， 合并图片
    count1 = 0
    count2 = 0
    out_put_path_random = os.path.join(out_put_path, randomstr)
    for res in results:
        if res is not None and res[1] is not None:
            if len(res[1]) == 0:
                count1 += 1
                print(res[0], "No pic paired")

            if len(res[1]) == 0:
                not_paired_save_path = out_put_path_random+'_not_paired'
                if not os.path.exists(not_paired_save_path):
                    os.makedirs(not_paired_save_path)
                merge_file_paths = []
                for index, value in enumerate(res[2]):
                    merge_file_paths.append(value[0].file_path)
                _imgs_merge(res[0], merge_file_paths, not_paired_save_path, os.path.basename(res[0]))
            elif len(res[1]) != 0:
                count2 += 1
                if not os.path.exists(out_put_path_random):
                    os.makedirs(out_put_path_random)
                # print('out' + out_put_path_random)
                merge_file_paths = []
                for index, value in enumerate(res[1]):
                    merge_file_paths.append(value[0].file_path)
                print(res[0], "paired with {} pictures".format(len(merge_file_paths)))
                _imgs_merge(res[0], merge_file_paths, out_put_path_random, os.path.basename(res[0]))
    print('has {} paired, {} unpaired'.format(count2, count1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compared marked pic with did generated pics')
    parser.add_argument('--marked', help='marked_pic_path', default='/Users/guorong/Documents/office/CCB_marked_data/data')
    parser.add_argument('--did', help='did_pic_path', default='/Users/guorong/Documents/office/pic_from_did/data/20190315-160319-420')
    parser.add_argument('--output', help='out_put_path', default='/Users/guorong/Documents/vipsdk/output')
    parser.add_argument('--faceserver', help='face server address', default='172.16.10.1:8096')
    parser.add_argument('--tracknum', help='calculate track num', type=int, default=0)
    parser.add_argument('--fluct', help='the time fluctuating to find face', default=10)
    args = parser.parse_args()
    marked_pic_path = args.marked
    did_pic_path = args.did
    out_put_path = args.output
    face_server = args.faceserver
    fluct = args.fluct
    track_num = None if args.tracknum == 0 else args.tracknum-1
    compare_marked_with_output(marked_pic_path, did_pic_path, out_put_path, face_server, track_num, fluct)

