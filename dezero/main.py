import numpy as np

from core import Square, Exp, Variable, Add


def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)


xs = [Variable(np.array(2)), Variable(np.array(3))]
f = Add()

ys = f(xs)
y = ys[0]
print(y.data)