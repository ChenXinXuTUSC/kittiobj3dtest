import os.path as osp
import yaml
import easydict

stored = easydict.EasyDict()

def read(yaml_path: str):
    # 读取 YAML 文件
    with open(yaml_path, 'r', encoding='utf-8') as file:
        # 将 YAML 内容转换为字典
        data = yaml.safe_load(file)
    conf_name, _ = osp.splitext(yaml_path)
    stored[conf_name] = easydict.EasyDict(data)
    return stored[conf_name]
