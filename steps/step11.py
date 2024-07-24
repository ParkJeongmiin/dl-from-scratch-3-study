'''
가변 길이 입출력을 표현하기 위해 연산과 관련된 클래스가
리스트 형태로 입력을 받으면 리스트 형태로 Variable 객체를 활용하도록 변경하였습니다.

<단점>
연산 과정에서 사용자가 직접 리스트 형태로 입력해야 하고, 출력은 지정된 형태로만 받을 수 있습니다.
'''
import numpy as np

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
            x, y = f.input, f.output    # 함수의 입력과 출력을 가져온다.
            x.grad = f.backward(y.grad) # backward 메서드를 호출한다.

            if x.creator is not None:
                funcs.append(x.creator) # 하나 앞의 함수를 리스트에 추가한다.

class Function:
    def __call__(self, inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        
        self.inputs = inputs
        self.outputs = outputs

        return outputs
    
    def forward(self, in_data):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()


class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)     # 튜플로 반환
    

def as_array(x):
    '''
    Variable 클래스에서 0차원의 ndarray 인스턴스를 제곱하게되면 numpy.float64 형태로 변환된다.
    numpy.ndarray 형태로 변환 시켜줘야 한다.
    그래서 Function 클래스에서 순전파 결과를 Variable 클래스로 초기화하기 전에 as_array로 확인 + 변환을 진행한다.
    '''
    if np.isscalar(x):
        return np.array(x)
    return x


# main
xs = [Variable(np.array(2)), Variable(np.array(3))]
f = Add()
ys = f(xs)
y = ys[0]
print(y.data)