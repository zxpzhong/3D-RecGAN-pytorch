# -*- coding: utf-8 -*-

import os
import zipfile


def get_zip_file(input_path, result,except_dir_inner = [],except_file_inner = []):
    """
    对目录进行深度优先遍历
    :param input_path:文件夹路径
    :param result:返回文件列表
    :param except_dir_inner:除去文件夹列表
    :param  except_file_inner:除去文件列表
    :return:
    """
    files = os.listdir(input_path)
    for file in files:
        if os.path.isdir(input_path + '/' + file):
            if not file in except_dir_inner:
                get_zip_file(input_path + '/' + file, result)
        elif not file in except_file_inner:
            # 后缀
            try:
                if file.split('.')[-1] in except_file_inner:
                    continue
            except:
                pass
            result.append(input_path + '/' + file)


def zip_file_path(input_path, output_path, output_name,except_dir = [],except_file = []):
    """
    压缩文件
    :param input_path: 压缩的文件夹路径
    :param output_path: 解压（输出）的路径
    :param output_name: 压缩包名称
    :param except_dir:除去文件夹列表
    :param  except_file:除去文件列表
    :return:
    """
    f = zipfile.ZipFile(output_path + '/' + output_name, 'w', zipfile.ZIP_DEFLATED)
    filelists = []
    get_zip_file(input_path, filelists,except_dir_inner = except_dir,except_file_inner = except_file)
    for file in filelists:
        f.write(file)
    # 调用了close方法才会保证完成压缩
    f.close()
    return output_path + r"/" + output_name


class ZIPCODE():
    def __init__(self,target_path = './',target_name = 'test.zip',source_path = './',except_dir = [],except_file = []):
        zip_file_path(source_path, target_path, target_name, except_dir, except_file)


