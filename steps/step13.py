'''
가변 길이 인수의 대응하는 Backward를 구현하였습니다.
입력이 2개라 가정하고 Add 클래스의 backward 메서드 출력이 2개가 되도록 구현하였습니다.
기존의 Variable 클래스에서 backward 메서드는 입력이 1개인 경우에 맞춰 구현되어 있어
여러 입력에서 동작하도록 구현하였습니다.
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
                x.grad = gx

                if x.creator is not None:
                    funcs.append(x.creator)


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


x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

z = add(square(x), square(y))       # z = x^2 + y^2
z.backward()

print("z = ", z.data)
print("x grad = ", x.grad)
print("y grad = ", y.grad)