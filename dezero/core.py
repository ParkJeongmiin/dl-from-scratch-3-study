import weakref
import contextlib
import numpy as np


# ================================================================
# Config
# ================================================================
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
    return using_config('enable_backprop', False)

# ================================================================
# Variable / Function
# ================================================================
class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0     # 세대 수를 기록하는 변수

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1   # 세대를 기록한다.(부모 세대 + 1)

    '''
    # 재귀를 이용한 구현
    def backward(self):
        f = self.creator                    # 1. 함수를 가져온다.
        if f is not None:
            x = f.input                     # 2. 함수의 입력을 가져온다.
            x.grad = f.backward(self.grad)  # 3. 함수의 backward 메서드를 호출한다.
            x.backward()                    # 하나 앞 변수의 backward 메서드를 호출한다.(재귀)
    '''

    # 반복문을 이용한 backward 구현
    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)     # main에서 y.grad를 직접 지정해주는 코드를 간소화하기 위해

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
        
        while funcs:
            f = funcs.pop()             # 함수를 가져온다.
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)
            
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None             # y는 약한 참조(weakref), 중간 변수의 미분값을 모두 None으로 설정

    def cleargrad(self):
        self.grad = None


def as_array(x):
    '''
    Variable 클래스에서 0차원의 ndarray 인스턴스를 제곱하게되면 numpy.float64 형태로 변환된다.
    numpy.ndarray 형태로 변환 시켜줘야 한다.
    그래서 Function 클래스에서 순전파 결과를 Variable 클래스로 초기화하기 전에 as_array로 확인 + 변환을 진행한다.
    '''
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
    
    def forward(self, in_data):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()


# ================================================================
# 연산자 오버로드
# ================================================================
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


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    

def exp(x):
    return Exp()(x)
    

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy
    

def add(x0, x1):
    return Add()(x0, x1)