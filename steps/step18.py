import weakref
import contextlib
import numpy as np


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    """
    using_config 함수를 편하게 사용하기 위한 편의함수
    """
    return using_config('enable_backprop', False)


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0     # 세대 수를 기록하는 변수

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1   # 세대를 기록한다.(부모 세대 + 1)

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)         # funcs 리스트에 같은 함수를 중복으로 추가하지 않기 위해
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):        # f.inputs[i]와 gxs[i] 서로 대응 관계에 있다.
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)
            
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None     # y는 약한 참조(weakref), 중간 변수의 미분값을 모두 None으로 설정

    def cleargrad(self):
        self.grad = None


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:                                          # 역전파 시에만 사용되는 로직들을 묶어 역전파 활성/비활성 모드로 관리할 수 있도록 합니다.
            self.generation = max([x.generation for x in inputs])           # 세대 설정 : 역전파 시 노드를 따라가는 순서를 정하는데 사용
            for output in outputs:
                output.set_creator(self)                                    # 계산들의 연결 설정 : 역전파 시 어떤 계산과 연결되었는지 확인하는데 사용

            self.inputs = inputs                                            # 순전파 결과를 저장하는 로직
            self.outputs = [weakref.ref(output) for output in outputs]      # 함수의 출력값을 약한 참조

        return outputs if len(outputs) > 1 else outputs[0]
    

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

def square(x):
    return Square()(x)


# with 블록 내에서는 순전파 코드만 실행된다.
with using_config('enable_backprop', False):
    x = Variable(np.array(2.0))
    y = square(x)

# 편의함수를 이용해 간편하게 순전파 영역을 지정할 수 있다.
with no_grad():
    
    x = Variable(np.array(2.0))
    y = square(x)