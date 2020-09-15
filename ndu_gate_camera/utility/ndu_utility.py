from re import search
from os import path, listdir
from platform import system
from importlib import util
from logging import getLogger
from inspect import getmembers, isclass

log = getLogger("service")


class NDUUtility:
    # Buffer for connectors/converters
    # key - class name
    # value - loaded class
    loaded_extensions = {}
    loaded_runners = {}

    @staticmethod
    def check_and_import(extension_type, module_name):
        if NDUUtility.loaded_extensions.get(extension_type + module_name) is None:

            file_dir = path.dirname(path.dirname(__file__))

            if system() == "Windows":
                extensions_paths = (path.abspath(file_dir + '/connectors/'.replace('/', path.sep) + extension_type.lower()),
                                    path.abspath(file_dir + '/../use_cases/'.replace('/', path.sep) + extension_type.lower()))
            else:
                extensions_paths = (
                    path.abspath(file_dir + '/connectors/'.replace('/', path.sep) + extension_type.lower()),
                    '/var/lib/thingsboard_gateway/extensions/'.replace('/', path.sep) + extension_type.lower(),
                    path.abspath(file_dir + '/../use_cases/'.replace('/', path.sep) + extension_type.lower()))
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
                                    log.debug(str(module))
                                    module_spec.loader.exec_module(module)
                                    for extension_class in getmembers(module, isclass):
                                        if module_name in extension_class:
                                            log.debug("Import %s from %s.", module_name, extension_path)
                                            # Save class into buffer
                                            NDUUtility.loaded_extensions[extension_type + module_name] = extension_class[1]
                                            return extension_class[1]
                                except ImportError:
                                    continue
                    else:
                        log.error("Import %s failed, path %s doesn't exist", module_name, extension_path)
            except Exception as e:
                log.exception(e)
        else:
            log.debug("Class %s found in TBUtility buffer.", module_name)
            return NDUUtility.loaded_extensions[extension_type + module_name]

    @staticmethod
    def check_and_import_all_runners():

        file_dir = path.dirname(path.dirname(__file__))

        if system() == "Windows":
            extensions_paths = (path.abspath(file_dir + '/../use_cases/'.replace('/', path.sep)),
                                path.abspath(file_dir + '/extensions/'.replace('/', path.sep)))
        else:
            extensions_paths = (
                path.abspath(file_dir + '/../use_cases/'.replace('/', path.sep)),
                '/var/lib/thingsboard_gateway/extensions/'.replace('/', path.sep),
                path.abspath(file_dir + '/extensions/'.replace('/', path.sep)))

        try:
            for extension_path in extensions_paths:
                if path.exists(extension_path):
                    for file in listdir(extension_path):
                        if not file.startswith('__') and file.endswith('.py'):
                            try:
                                module_spec = util.spec_from_file_location("test", extension_path + path.sep + file)
                                log.debug(module_spec)

                                if module_spec is None:
                                    log.error('Module: %s not found', "test")
                                    continue

                                module = util.module_from_spec(module_spec)
                                log.debug(str(module))
                                module_spec.loader.exec_module(module)
                                for extension_class in getmembers(module, isclass):
                                    log.debug("extension_class %s ", extension_class)
                                    # if module_name in extension_class:
                                    #     log.debug("Import %s from %s.", module_name, extension_path)
                                    #     NDUUtility.loaded_extensions[extension_type + module_name] = extension_class[1]
                                    #     return extension_class[1]
                            except ImportError:
                                continue
                else:
                    log.error("Import %s failed, path %s doesn't exist", "test", extension_path)
        except Exception as e:
            log.exception(e)

        return NDUUtility.loaded_runners;
