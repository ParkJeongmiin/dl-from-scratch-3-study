import numpy as np

from core import *


# with 블록 내에서는 순전파 코드만 실행된다.
with using_config('enable_backprop', False):
    x = Variable(np.array(2.0))
    y = square(x)

# 편의함수를 이용해 간편하게 순전파 영역을 지정할 수 있다.
with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)