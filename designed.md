# 框架目的

## DDP 分布式训练

```txt
Trainer.run():
	# mp.spawn into ddp mode
		ddp_entry():
			ddp_setup()
			train()
			ddp_cleanup()
```

## 包装方式

主要来说框架通过读取配置文件中，相应组件的字符串名称，然后去注册表里面通过该名称拿对应的类型构造函数，构造完所有的组件之后，开始并行分布式训练。

# 主要结构

```txt
## Model 模型
	存放相关的模型代码
## Dataset 数据集
	数据集适配代码，用户必须知道自己实现的数据集代码和哪些模型要求的输入是匹配的。同时数据集可以实现自定义的 `collate_fn` 来自定义进入 `torch.dataloader` 之前如何封装一个批次的样本，比如每个样本具有不同的数据长度那么默认的 `collate_fn` 就无法使用 `torch.Tensor` 完成转换，需要自定义 `collate_fn` 完成样本批中不同长度样本填充到相同最大长度。另外主框架流程中也不应该关心 `dataloder` 具体返回了什么值，应该如何使用，而应该只是传递这个值给模型，由模型来自行取用 `batch` 批次样本中的值，但是模型不知道自己所在的 `device` ，所以检测 `batch` 中类型是 `torch.Tensor` 的值并转移到模型对应的 `device` 上，这个是框架的责任。
## Metric 评估指标
	训练过程中模型调用的评估指标，模型将所有输出作为依赖参数传递给评估指标函数，在这里负责一些日志的记录，比如 TensorBoard 的可视化图像日志记录
## LOSS 损失函数
	框架的用户应该针对其所提供的训练模型，实现相应的损失计算函数，取用数据加载器的数据和模型的输出（或其中的一部分）完成实现。损失函数应该相对于模型实现，但有时候用户可能会有一些其他自定义的损失过程计算，涉及到原始的数据。
## Conf 配置文件
	指定一次运行时，配套的 【模型】+【数据集】+【评估指标】+【损失函数】
```

# 踩坑

## 多进程全局日志器的创建
> 但不幸的是，如果提前创建好日志器，比如使用了 `logging.logger` 等模块，那么日志器对象会持有对日志文件的锁，而持有锁的对象是无法在 `mp.spawn` 生成多进程组的时候序列化为对象传递给子进程的，因此我不得不将日志器的创建放在每个子进程被初始化之后，每个进程判断自己是否需要创建日志器（通过判断 `rank` 来实现）。

```python
if self.world_rank == 0:
	self.loggertfx = utils.logger.LoggerTXTFX(
		self.log_alldir, name=self.log_exname
	)
else:
	self.loggertfx = None
```

只有主进程才拥有日志器。

## mp.spawn 不接受双下划线的属性

我就是有很多奇怪的想法突然想尝试，想学别人那些库，把不想给外界访问的方法用双下划线装饰，结果 `mp.spawn` 也不能访问被双下划线修饰的函数……


## nn.CrossEntropyLoss 不支持非连续标签

SemanticKITTI 的每个类别的标签值不是连续的，就是说背景类是 0 ，但是后面的类不是依次递增的，所以 40 多个类别但是最大整型标签索引去到了 255 。计算交叉熵损失函数的时候 cuda 库断言报错说输入的类别索引大于了类别数的索引：
```txt
/opt/conda/conda-bld/pytorch_1659484809662/work/aten/src/ATen/native/cuda/NLLLoss2d.cu:103: nll_loss2d_forward_kernel: block: [1,0,0], thread: [128,0,0] Assertion `t >= 0 && t < n_classes` failed.
```

PyTorch 的 CrossEntropyLoss (CLE)  有多种调用方式，取决于你的输入数据的形状和你的需求。  主要区别在于目标标签 (target) 的形状以及是否需要指定权重。

基本用法：

```python
loss = nn.CrossEntropyLoss()(input, target)
```

input (Tensor):  模型的输出，形状通常为 (N, C)，其中 N 是批次大小，C 是类别数。  每个元素表示对应样本属于每个类别的概率或logit值(取决于你模型的输出层是否有softmax)。 不包含softmax。

target (Tensor):  目标标签，形状通常为 (N,)，包含每个样本的类别索引（整数），从 0 到 C-1。  必须是长整数类型(torch.long)。

## 损失函数权重和数据集类别不配对

尝试 SemanticKITTI 数据集的时候，发现 KITTIObjec3D 的数据集使用的损失函数不能给 SemanticKITTI 用，因为 KITTI 只有不到十个类别，所以可以手写权重，但是 SemanticKITTI 有 40 多个类别，光是权重数量就不一样，所以不能复用之前的损失函数了……

然后我发现不仅损失函数要改，模型的输出通道也需要对应跟着数据集的类别数修改。

另外对于场景比较大的数据集，SemanticKITTI 有时候只能投影到一部分，可能只有单个类别，

