import numpy as np

from core import *


for i  in range(10):
    x = Variable(np.random.randn(10000))        # 거대한 데이터
    y = square(square(square(x)))               # 복잡한 계산
    