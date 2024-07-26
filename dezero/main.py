import numpy as np

from core import *


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