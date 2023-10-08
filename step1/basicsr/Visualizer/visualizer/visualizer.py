from bytecode import Bytecode, Instr
import pdb

class get_local(object):
    """
    Decorator that takes a list of variable names as argument. Everytime
    the decorated function is called, the final states of the listed
    variables are logged and can be read any time during code execution.    
    """
    cache = {}
    is_activate = False

    def __init__(self, varnames):
        self.varnames = varnames 



    def __call__(self, func):
        if not type(self).is_activate: # type(self) = <class 'Visualizer.visualizer.visualizer.get_local'>
            return func

        type(self).cache[func.__qualname__] = []        
        c = Bytecode.from_code(func.__code__)
        extra_code = [
            Instr('STORE_FAST', '_res')
        ]+[
            Instr('LOAD_FAST', name) for name in self.varnames
        ]+[
            Instr('BUILD_TUPLE', len(self.varnames)),
            Instr('STORE_FAST', '_value'),
            Instr('LOAD_FAST', '_res'),
            Instr('LOAD_FAST', '_value'),
            Instr('BUILD_TUPLE', 2),
            Instr('STORE_FAST', '_result_tuple'),
            Instr('LOAD_FAST', '_result_tuple'),
        ]        
        c[-1:-1] = extra_code
        func.__code__ = c.to_code()


        def wrapper(*args, **kwargs):
            res, values = func(*args, **kwargs) #感觉res是输入， values是要取出的局部变量,运行后跳转到Attention_dec中执行 func是Attention_dec.forward
            # res.shape[1,16,512] q: values[0].shape[1,8,16,64], v: values[1].shape[1,8,16,64]
            #type(self).cache[func.__qualname__].append(values.detach().cpu().numpy()) # 以字典形式存储
            type(self).cache[func.__qualname__].append(values) # 以字典形式存储
            return res
        return wrapper

    @classmethod
    def clear(cls):
        for key in cls.cache.keys():
            cls.cache[key] = []

    @classmethod
    def activate(cls):
        cls.is_activate = True


