#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import argparse
import json
import mysql.connector
import os
import numpy as np
from urllib.request import urlopen
import cv2
import matplotlib.pyplot as plt

def _write_to_db(data, cursor, cnx):
    """
    将数据写入db, 重复的request id不会被写入
    :param data:
    :return:
    """
    # print(data)

    cursor.execute(
        "SELECT request_id, COUNT(*) FROM match_image_score WHERE request_id = %s GROUP BY request_id",
        (data['request_id'],)
    )
    # gets the number of rows affected by the command executed
    row_count = cursor.rowcount
    # print("number of affected rows: {}".format(row_count))
    if row_count == 0:
        # print("It Does Not Exist")
        add_record = ("insert into match_image_score"
                      "(matched_image, score, temp_regis_image, temp_regis_image_score, "
                      "history_regis_image, history_regis_image_score, request_id)"
                      "values (%s, %s, %s, %s, %s, %s, %s)")


        try:
            cursor.execute(add_record, (
                data['matched_image'],
                data['score'],
                data['temp_regis_image'],
                data['temp_regis_image_score'],
                data['history_regis_image'],
                data['history_regis_image_score'],
                data['request_id']
            ))
            cnx.commit()
        except Exception as e:
            print(repr(e))


def write_log(log_path):
    """
    读日志， 获得partition 日志
    获取 image_url, match_url, score, 写入数据库
    :param log_path:
    :return:
    """
    cnx = mysql.connector.connect(user='rguo',
                                  password='rguo',
                                  host='172.16.10.1',
                                  database='mvp_test_data'
                                  )
    cursor = cnx.cursor(buffered=True)
    try:
        with open(log_path, 'r', encoding='utf-8') as log:
            print('file opened')
            for line in log:
                # print(line)
                if 'partition' in line:
                    flag = True
                    rest_line = line.split('data:')[1]
                    rest_json = json.loads(rest_line)
                    data = {}

                    data['score'] = rest_json['match_score']
                    data['request_id'] = rest_json['request_id']
                    index = rest_json['match_photo_index']
                    if rest_json['photos'] is not None:
                        data['matched_image'] = rest_json['photos'][index]['url']
                        data['matched_image_quality'] = rest_json['photos'][index]['quality']
                    else:
                        flag = False

                    if rest_json['process_context']['temp_res'] is None and rest_json['process_context']['history_res'] is None:
                        flag = False
                    else:
                        data['temp_regis_image'] = None if (rest_json['process_context']['temp_res'] is None) else \
                            (rest_json['process_context']['temp_res']['top_user']['image_url'])
                        data['temp_regis_image_score'] = 0 if (rest_json['process_context']['temp_res'] is None) else \
                            (rest_json['process_context']['temp_res']['top_user']['score'])
                        data['history_regis_image'] = None if (rest_json['process_context']['history_res'] is None) else \
                            (rest_json['process_context']['history_res']['top_user']['image_url'])
                        data['history_regis_image_score'] = 0 if (rest_json['process_context']['history_res'] is None) else \
                            (rest_json['process_context']['history_res']['top_user']['score'])
                    if flag:
                        _write_to_db(data, cursor, cnx)
    finally:
        cursor.close()
        cnx.close()


def _cancatenate_compared_images(data):
    """
    获取 image_url, match_url, score, 写入数据库
    查询的时候，输入id 范围，返回一张大图片 n*2, 图片左上角写上score
    :param data: {
    "compare_pic":"url",
    "compare_score":0,
    "temp_pic":"",
    "temp_score: 0,
    "history_pic":"",
    "history_score":0
    }
    :return:
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (30, 80)
    fontScale = 2
    fontColor = (255, 255, 255)
    lineType = 2

    compare_pic = _url_to_image(data['compare_pic'])
    temp_pic = _url_to_image(data['temp_pic'])
    history_pic = _url_to_image(data['history_pic'])
    cv2.putText(
        compare_pic,
        str(data['compare_score']),
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType
        )
    cv2.putText(
        temp_pic,
        str(data['temp_score']),
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType
    )
    cv2.putText(
        history_pic,
        str(data['history_score']),
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType
    )
    image_list = [compare_pic, temp_pic, history_pic]
    res_img = None
    for idx, img in enumerate(image_list):
        _img = cv2.resize(img, (int(img.shape[1] * 400 / img.shape[0]), 400), interpolation=cv2.INTER_CUBIC)
        if idx == 0:
            res_img = _img
        else:
            res_img = np.concatenate((res_img, _img), axis=1)
    return res_img



def _url_to_image(url):
    if not url:
        image = np.zeros((400, 400, 3), np.uint8)
    else:
        try:
            resp = urlopen(url)
            image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        except Exception as e:
            print(url)
            print(repr(e))
            image = np.zeros((400, 400, 3), np.uint8)

    return image

def _get_log_list(min_score, max_score, cursor):
    """
    通过给最大值和最小score 查询数据, 大于100条则报日志，缩小范围
    :param min_score:
    :param max_score:
    :return:
    """
    search_records = (" select matched_image, score, "
                      "temp_regis_image, temp_regis_image_score,"
                      "history_regis_image, history_regis_image_score"
                      " from match_image_score "
                      "where score < %s and score >= %s"
                      " order by score;")
    # print(search_records)
    cursor.execute(search_records, (max_score, min_score))
    result = cursor.fetchall()
    if len(result) > 1000:
        print('selected records is larger than 1000 :{}. '.format(len(result)))

    return list(result)

def get_image_list(min_score, max_score, output):
    cnx = mysql.connector.connect(user='rguo',
                                  password='rguo',
                                  host='172.16.10.1',
                                  database='mvp_test_data'
                                  )
    cursor = cnx.cursor()
    records = _get_log_list(min_score=min_score, max_score=max_score, cursor=cursor)
    if records is None or len(records) == 0 :
        print('no records in score range has been found')
    else:
        # print(records)
        keys_list = ('compare_pic', 'compare_score', 'temp_pic', 'temp_score',
                     'history_pic', 'history_score')
        records_list = list(dict(zip(keys_list, x)) for x in records)
        # print(records_list)
        big_pic = None
        bucket_num =10
        list_len = len(records_list)
        print(list_len)
        num1 = int(list_len/bucket_num)+1
        print(num1)
        for i in range(0, num1):
            for idx, data in enumerate(records_list[bucket_num*i:bucket_num*i+bucket_num]):
                print(data)
                if idx == 0:
                    big_pic = _cancatenate_compared_images(data=data)
                else:
                    big_pic = np.concatenate((big_pic, _cancatenate_compared_images(data=data)))
            output_file = os.path.join(output, '_'.join([str(min_score), str(max_score), str(i)])+'.png')
            cv2.imwrite(output_file, big_pic)
            print('image has been save to path {}'.format(output_file))
        # cv2.imshow(output, big_pic)

        # cv2.waitKey(0)  # waits until a key is pressed
        # cv2.destroyAllWindows()  # destroys the window showing image


def draw_score_histgram(score_type):
    """
    画出质量的直方图
    :param score_type: 三个值  score, temp_regis_image_score, history_regis_image_score
    :return:
    """
    cnx = mysql.connector.connect(user='rguo',
                                  password='rguo',
                                  host='172.16.10.1',
                                  database='mvp_test_data'
                                  )
    cursor = cnx.cursor(buffered=True)
    search_records = ("select {} from match_image_score".format(score_type))
    try:
        cursor.execute(search_records)
        result = cursor.fetchall()
        result_conv = [x[0] for x in result]
    finally:
        cursor.close()
        cnx.close()

    rdata = plt.hist(result_conv, density=False, bins=5)
    print(rdata)
    plt.xlabel(score_type)
    plt.ylabel('number')
    plt.title('score hist')
    # plt.axis([0, 100, 0, 5])
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    """
    脚本读取common-push服务的日志，产出图片来查找阈值。
    阈值1以上，图片比对全部正确
    阈值2以下，图片比对全部失败
    """
    parser = argparse.ArgumentParser(description='get image registered, compared pic and score')
    parser.add_argument('--log', help='log path', default='/Users/guorong/Documents/nome/checkthreshold/common-push.log')
    parser.add_argument('--minscore', help='搜索的最小score', type=int, default=90)
    parser.add_argument('--maxscore', help='搜索的最大score', type=int, default=92)
    parser.add_argument('--output', help='文件输出地址', type=str, default='/Users/guorong/Documents/nome/gpic/')
    args = parser.parse_args()
    min_score = args.minscore
    max_score = args.maxscore
    out_put = args.output
    log_path = args.log
    if os.path.exists(log_path):
        print('log_path is {}'.format(log_path))
    # write_log(log_path)
    # draw_score_histgram('temp_regis_image_score')
    get_image_list(min_score, max_score, out_put)

