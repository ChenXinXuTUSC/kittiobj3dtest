class Metric:
    def __init__(self):
        pass

    def best(self):
        pass

    def mean(self):
        pass

    def reset(self):
        pass
    
    def mct(self,
        data, pred, gdth
    ):
        raise NotImplementedError("derived mct not defined")

    # console and tensorboard log
    # print the last computed metric
    def log(self,
        
    ):
        raise NotImplementedError("derived log not defined")
