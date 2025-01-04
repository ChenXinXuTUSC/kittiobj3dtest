import os
import os.path as osp
import logging


def get_logger(logger_name: str, to_file: bool=False, log_file: str=""):
    # generated by DeepSeek


    logging.addLevelName(logging.DEBUG, "DBUG")
    logging.addLevelName(logging.INFO, "INFO")
    logging.addLevelName(logging.WARNING, "WARN")
    logging.addLevelName(logging.ERROR, "ERRO")
    logging.addLevelName(logging.CRITICAL, "FATL")

    # 创建一个日志记录器
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # 设置日志级别

    # 创建一个格式化器，定义日志的输出格式
    formatter = logging.Formatter(
        # '[%(asctime)s|%(name)s|%(levelname)s] %(message)s',
        '[%(asctime)s|%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 创建一个控制台处理器，并将日志级别设置为 DEBUG
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 创建一个文件处理器，并将日志级别设置为 DEBUG

    if to_file:
        log_dir = osp.dirname(log_file)
        if not osp.exists(log_dir):
            os.makedirs(log_dir)
        if not osp.exists(log_dir):
            raise FileNotFoundError
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


    # 将格式化器添加到处理器中

    # 将处理器添加到日志记录器中

    return logger

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
class LoggerTXTFX:
    def __init__(self, root: str, name: str):
        self.__start_time = datetime.now().strftime("%Y%m%d%H%M%S")
        self.__root = osp.join(root, name, self.__start_time)


        self.__root_tfx = osp.join(self.__root, "tfx")
        if not osp.exists(self.__root_tfx):
            os.makedirs(self.__root_tfx, exist_ok=True)
        self.__tfx_logger = SummaryWriter(self.__root_tfx)

        self.__root_txt = osp.join(self.__root, "txt")
        if not osp.exists(self.__root_txt):
            os.makedirs(self.__root_txt, exist_ok=True)
        self.__txt_logger = get_logger(
            name, True,
            osp.join(self.__root_txt, "run.log")
        )

    @property
    def txt(self):
        """返回文本日志记录器"""
        return self.__txt_logger

    @property
    def tfx(self):
        """返回 TensorBoard 日志记录器"""
        return self.__tfx_logger
    
    def get_root(self):
        return self.__root
