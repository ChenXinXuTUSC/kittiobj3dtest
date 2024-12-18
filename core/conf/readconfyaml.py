import yaml
import easydict


def read(yml_path: str):
    # 读取 YAML 文件
    with open(yml_path, 'r', encoding='utf-8') as file:
        # 将 YAML 内容转换为字典
        data = yaml.safe_load(file)
    return easydict.EasyDict(data)
