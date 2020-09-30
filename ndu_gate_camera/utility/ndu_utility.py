import sys
import re
from os import path, listdir
from platform import system
from importlib import util
from logging import getLogger
from inspect import getmembers, isclass, isfunction

log = getLogger("service")


class NDUUtility:
    # Buffer for connectors/converters
    # key - class name
    # value - loaded class
    loaded_runners = {}

    @staticmethod
    def check_and_import(extension_type, module_name):
        if NDUUtility.loaded_runners.get(extension_type + module_name) is None:
            file_dir = path.dirname(path.dirname(__file__))
            if system() == "Windows":
                extensions_paths = (path.abspath(file_dir + '/runners/'.replace('/', path.sep) + extension_type.lower()))
            else:
                extensions_paths = ('/var/lib/thingsboard_gateway/extensions/'.replace('/', path.sep) + extension_type.lower(),
                                    '/var/lib/ndu_gate/runners/'.replace('/', path.sep) + extension_type.lower(),
                                    path.abspath(file_dir + '/runners/'.replace('/', path.sep) + extension_type.lower()))

            if NDUUtility.is_debug_mode():
                extensions_paths_list = list(extensions_paths)
                extensions_paths_list.append(path.abspath(file_dir + '/../runners/'.replace('/', path.sep) + extension_type.lower()))
                extensions_paths = tuple(extensions_paths_list)

            try:
                for extension_path in extensions_paths:
                    if path.exists(extension_path):
                        for file in listdir(extension_path):
                            if not file.startswith('__') and file.endswith('.py'):
                                try:
                                    module_spec = util.spec_from_file_location(module_name, extension_path + path.sep + file)
                                    log.debug(module_spec)

                                    if module_spec is None:
                                        log.error('Module: %s not found', module_name)
                                        continue

                                    module = util.module_from_spec(module_spec)
                                    log.info(str(module))
                                    module_spec.loader.exec_module(module)
                                    for extension_class in getmembers(module, isclass):
                                        if module_name in extension_class:
                                            log.info("Import %s from %s", module_name, extension_path)
                                            NDUUtility.loaded_runners[extension_type + module_name] = extension_class[1]
                                            log.info("Total runners : %s", len(NDUUtility.loaded_runners))
                                            return extension_class[1]
                                except Exception as ie:
                                    log.error(ie)
                                    continue
                    else:
                        log.error("Import %s failed, path %s doesn't exist", module_name, extension_path)
            except Exception as e:
                log.exception(e)
        else:
            log.info("Class %s found in NDUUtility buffer.", module_name)
            return NDUUtility.loaded_runners[extension_type + module_name]

    @staticmethod
    def get_methods(cls_):
        methods = getmembers(cls_, isfunction)
        return dict(methods)

    @staticmethod
    def has_method(cls_, name):
        methods = getmembers(cls_, isfunction)
        return name in dict(methods)

    @staticmethod
    def install_package(package, version="upgrade"):
        from sys import executable
        from subprocess import check_call, CalledProcessError
        result = False
        if version.lower() == "upgrade":
            try:
                result = check_call([executable, "-m", "pip", "install", package, "--upgrade", "--user"])
            except CalledProcessError:
                try:
                    result = check_call([executable, "-m", "pip", "install", package, "--upgrade"])
                except Exception as e:
                    log.error(e)
        else:
            from pkg_resources import get_distribution
            current_package_version = None
            try:
                current_package_version = get_distribution(package)
            except Exception as e:
                log.error(e)
                pass
            if current_package_version is None or current_package_version != version:
                installation_sign = "==" if ">=" not in version else ""
                try:
                    result = check_call([executable, "-m", "pip", "install", package + installation_sign + version, "--user"])
                except CalledProcessError:
                    try:
                        result = check_call([executable, "-m", "pip", "install", package + installation_sign + version])
                    except Exception as e:
                        log.error(e)

        return result

    @staticmethod
    def is_url_valid(url) -> bool:
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        return re.match(regex, url) is not None

    @staticmethod
    def is_debug_mode():
        get_trace = getattr(sys, 'gettrace', None)
        return get_trace is not None
