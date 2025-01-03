class Registry:
    def __init__(self, name):
        """
        初始化一个注册表。

        Args:
            name (str): 注册表的名称，用于标识注册表的用途。
        """
        self._name = name
        self._registry = {}  # 用于存储注册的类

    def register(self, cls):
        """
        装饰器函数，用于将类注册到注册表中。

        Args:
            cls: 需要注册的类。

        Returns:
            返回原始的类。
        """
        # 将类名作为键，类本身作为值注册到哈希表中
        self._registry[cls.__name__] = cls
        return cls

    def get(self, name):
        """
        根据名称从注册表中获取类。

        Args:
            name (str): 类的名称。

        Returns:
            返回注册的类，如果未找到则返回 None。
        """
        return self._registry.get(name)
    
    def __getitem__(self, name):
        """
        根据名称从注册表中获取类。
        Args:
            name (str): 类的名称。
        Returns:
            返回注册的类，如果未找到则返回 None。
        """
        return self._registry.get(name)

    def __contains__(self, name):
        """
        检查某个类是否已经注册。

        Args:
            name (str): 类的名称。

        Returns:
            bool: 如果类已注册则返回 True，否则返回 False。
        """
        return name in self._registry

    def __str__(self):
        """
        返回注册表的字符串表示。

        Returns:
            str: 注册表的名称和内容。
        """
        return f"Registry(name={self._name}, registry={self._registry})"

    @property
    def name(self):
        """
        返回注册表的名称。

        Returns:
            str: 注册表的名称。
        """
        return self._name

    @property
    def registry(self):
        """
        返回注册表的完整内容。

        Returns:
            dict: 注册表的完整内容。
        """
        return self._registry
