import numpy as np

from core import *


x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

z = add(square(x), square(y))
z.backward()

print("z = ", z.data)
print("x grad = ", x.grad)
print("y grad = ", y.grad)