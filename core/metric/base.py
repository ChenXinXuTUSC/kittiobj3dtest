class Metric:
    def __init__(self):
        pass

    def best(self) -> bool:
        # best 应该返回根据目前积累的指标数据计算出的指标
        # 是否优于之前计算过的最好指标
        raise NotImplementedError("metric best not defined")

    def mean(self):
        raise NotImplementedError("metric mean not defined")

    def reset(self):
        raise NotImplementedError("metric reset not defined")
    
    def mct(self,
        data, pred, gdth
    ):
        raise NotImplementedError("derived mct not defined")

    # console and tensorboard log
    # print the last computed metric
    def log(self,
        
    ):
        raise NotImplementedError("derived log not defined")
