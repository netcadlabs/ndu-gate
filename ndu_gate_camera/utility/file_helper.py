import codecs
import io
import os
import pathlib
from typing import AnyStr

from ndu_gate_camera.utility.ndu_utility import NDUUtility


def read_lines(filename, encoding="utf-8"):
    with io.open(filename, mode="r", encoding=encoding) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        return content


def write_all(filename, text, encoding="utf-8"):
    write_lines(filename, [text], encoding)


def write_lines(filename, lines, encoding="utf-8"):
    with codecs.open(filename, "w", encoding) as f:
        for item in lines:
            f.write("%s\n" % item)


def enumerate_files(dir_path, recursive=False, wildcard_pattern=None, case_insensitive=True):
    # if not recursive:
    #     from os import listdir
    #     from os.path import isfile, join
    #     if filter is None:
    #         for file_name_with_extension in [f for f in listdir(dir_path) if isfile(join(dir_path, f))]:
    #             yield os.path.join(dir_path, file_name_with_extension), file_name_with_extension
    #     else:
    #         for file_name_with_extension in [f for f in listdir(dir_path) if isfile(join(dir_path, f))]:
    #             if str_helper.wildcard(file_name_with_extension, filter, case_insensitive):
    #                 yield os.path.join(dir_path, file_name_with_extension), file_name_with_extension
    if wildcard_pattern is None:
        for root, sub_dirs, files in os.walk(dir_path):
            for fn in files:
                yield path_join(root, fn)
            if not recursive:
                break
    else:
        for root, sub_dirs, files in os.walk(dir_path):
            for fn in files:
                name = os.path.basename(fn)
                if NDUUtility.wildcard(name, wildcard_pattern, case_insensitive=case_insensitive):
                    yield fn
            if not recursive:
                break


def get_parent_dir_path(file_name):
    return str(pathlib.Path(file_name).parent.absolute())


def get_parent_dir_name(file_name):
    return str(pathlib.Path(file_name).parent.name)


def path_join(a: AnyStr, *paths: AnyStr) -> AnyStr:
    return os.path.join(a, *paths).replace("/", os.path.sep)
