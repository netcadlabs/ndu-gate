import codecs
import io
import os
import pathlib
from typing import AnyStr, Tuple
from shutil import copyfile

from ndu_gate_camera.utility import string_helper
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
    #             if string_helper.wildcard(file_name_with_extension, filter, case_insensitive):
    #                 yield os.path.join(dir_path, file_name_with_extension), file_name_with_extension
    if wildcard_pattern is None:
        for root, sub_dirs, files in os.walk(dir_path):
            for name in files:
                yield path_join(root, name)
            if not recursive:
                break
    else:
        for root, sub_dirs, files in os.walk(dir_path):
            for name in files:
                # name = os.path.basename(fn)
                if string_helper.wildcard(name, wildcard_pattern, case_insensitive=case_insensitive):
                    yield path_join(root, name)
            if not recursive:
                break


def append_line(filename, line, encoding="utf-8"):
    with codecs.open(filename, "a", encoding) as f:
        f.write("%s\n" % line)


def get_parent_dir_path(file_name):
    return str(pathlib.Path(file_name).parent.absolute())


def get_parent_dir_name(file_name):
    return str(pathlib.Path(file_name).parent.name)


def path_join(a: AnyStr, *paths: AnyStr) -> AnyStr:
    return os.path.join(a, *paths).replace("/", os.path.sep)


def create_dir(dir_name, parents=True, exist_ok=True):
    from pathlib import Path
    Path(dir_name).mkdir(parents=parents, exist_ok=exist_ok)


def copy_file(source, target):
    copyfile(source, target)


def get_file_name_extension(file_full_name) -> Tuple[AnyStr, AnyStr, AnyStr]:
    dir_name, file_name = os.path.split(file_full_name)
    name, extension = os.path.splitext(file_name)
    return dir_name, name, extension


def delete_file(file_name):
    os.remove(file_name)