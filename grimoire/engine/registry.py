class BaseModuleRegistry:
    def __init__(self, name=None) -> None:
        self._registry = {}
        self._name = self.__class__.__name__ if not name else name

    def _register_module(self,
                         module,
                         module_name = None) -> None:
        
        if module_name is None:
            module_name = module.__name__
        
        if isinstance(module_name, str):
            module_name = [module_name]
        
        for name in module_name:
            if name in self._registry:
                existed_module = self._registry[name]
                raise KeyError(f'{name} is already registered in {self._name} at {existed_module.__module__}')
        
        self._registry[name] = module
    
    def register_module(self, module=None, name=None):

        def _registry_decorator(module):
            self._register_module(module=module, module_name=name)
            return module

        return _registry_decorator


if __name__ == "__main__":
    registry = BaseModuleRegistry(name="ExampleRegistry")

    @registry.register_module()
    class A:
        pass

    @registry.register_module(name='BBB')
    class B:
        pass

    @registry.register_module(name='BBB')
    class B:
        pass

    print(registry._registry)

    # optimizers_registry = BaseModuleRegistry(name="optimizers_registry")
    # losses_registry = BaseModuleRegistry(name="losses_registry")

    # def register_torch_optimizers():
    #     torch_optimizers = []
    #     for module_name in dir(torch.optim):
    #         if module_name.startswith('__'):
    #             continue
    #         _optim = getattr(torch.optim, module_name)
    #         if inspect.isclass(_optim) and issubclass(_optim,
    #                                                 torch.optim.Optimizer):
    #             optimizers_registry.register_module(module=_optim)
    #             torch_optimizers.append(module_name)
    #     return torch_optimizers