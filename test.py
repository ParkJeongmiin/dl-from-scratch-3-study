import unittest
import numpy as np

from dezero.core import Square, Exp, Variable


def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)


def numerical_diff(f, x, eps=1e-4):
    '''
    (수치 미분으로 구한 결과)와 (역전파 결과)를 비교해
    테스트를 진행하려고 합니다.
    수치 미분의 결과를 얻기 위해 구현된 함수입니다.
    '''
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)
    
    
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)


    def test_gradient_check(self):
        x = Variable(np.random.random(1))   # random 입력값
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


'''
아래의 코드를 추가하면 터미널에서
python -m unittest test.py 대신에
python test.py 처럼 평소대로 실행시켜 테스트 모드로 실행할 수 있다.
'''
# unittest.main()