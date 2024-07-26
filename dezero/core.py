import numpy as np


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

    def set_creator(self, func):
        self.creator = func

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
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)     # main에서 y.grad를 직접 지정해주는 코드를 간소화하기 위해

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()             # 함수를 가져온다.
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                x.grad = gx

                if x.creator is not None:
                    funcs.append(x.creator)


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

        for output in outputs:
            output.set_creator(self)
        
        self.inputs = inputs
        self.outputs = outputs

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