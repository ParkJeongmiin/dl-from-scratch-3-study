'''
step 11에서는 Add 함수의 입력과 출력이 각각 리스트와 튜플로 한정되어 있습니다.
이것은 사용하는 사람의 입장에서는 불편함을 줄 수 있기 때문에
"함수를 호출할 때 원소를 낱개로 풀고, 출력을 튜플로 변경하여 출력하도록" 수정하였습니다.
'''
import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):                                #
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)                                  # 언팩을 통해 원소를 낱개로 전달
        if not isinstance(ys, tuple):                           # 튜플이 아닌 경우에 튜플로 변경해 반환
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]      #
    

class Add(Function):                                            #
    def forward(self, x0, x1):                                  #
        y = x0 + x1                                             # 입력받은 값을 그대로 사용하도록 변경
        return y                                                #
    

def add(x0, x1):
    return Add()(x0, x1)


x0 = Variable(np.array(2))
x1 = Variable(np.array(4))

y = add(x0, x1)
print(y.data)