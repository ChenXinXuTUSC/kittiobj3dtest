import os.path as osp
import toml
import easydict

stored = easydict.EasyDict()

def read(conf_path: str):
	# 读取 YAML 文件
	with open(conf_path, 'r', encoding='utf-8') as f:
		# 将 YAML 内容转换为字典
		data = toml.load(f)
	conf_name, _ = osp.splitext(conf_path)
	stored[conf_name] = easydict.EasyDict(data)
	return stored[conf_name]
