'''
지금까지 구현에서 문제점을 발견했습니다.

"동일한 변수를 반복해서 사용하면 미분값이 의도대로 나오지 않습니다."
원인은 Variable 클래스에서 backward 메서드를 구현하는 과정에서 
역전파로 전해지는 미분값이 출력에서 grad 인스턴스 변수에 초기화 되도록 구현되었기 때문입니다.
같은 변수를 사용하게 되면 미분값이 계속 덮어 써지면서 의도대로 동작하지 않게 됩니다.

이것을 해결하기 위해서 
기존 출력 쪽 grad 인스턴스가 처음 사용되었다면 바로 초기화 하고
앞에서 같은 변수가 사용되었다면 미분값을 더하도록 변경하였습니다. 
===========================================================================================
여기까지만 수정하면 또 다른 문제가 발생하게 됩니다.
"같은 변수를 서로 다른 수식에서 사용하게 되면 저장된 미분값이 누적되어 계산이 꼬일 수 있습니다."

grad 인스턴스를 초기화하는 메서드를 추가해 재사용 전에 초기화해 계산에 차질이 없도록 하였습니다.
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
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):        # f.inputs[i]와 gxs[i] 서로 대응 관계에 있다.
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    funcs.append(x.creator)

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

        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs

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


# 첫 번째 계산
x = Variable(np.array(3.0))
y = add(x, x)
y.backward()
print('x.grad', x.grad)

# 두 번째 계산(같은 x를 사용하여 다른 계산을 수행)
x.cleargrad()
y = add(add(x, x), x)
y.backward()
print('x.grad', x.grad)