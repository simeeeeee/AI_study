#Gradient descent실습 : Learning rate
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def f(x):
    return x**2 -4*x +6

NumberOfPoints=101
#x는 -5부터 5까지 101개의 등간격 값들로 구성된 배열
x = np.linspace(-5., 5, NumberOfPoints)
fx = f(x)

plt.plot(x,fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')
# plt.show()

# 배열 fx에서 최소값의 인덱스(index) 반환.
xid = np.argmin(fx)
# 최소가 되는 x값 (xopt) 구함
xopt = x[xid]
print(xopt, f(xopt))

plt.plot(x,fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')

plt.plot(xopt, f(xopt), 'xr')
# plt.show()

# 기울기
def grad_fx(x):
    return 2*x -4

# w보정
def steepest_descent(func, grad_func, x0, learning_rate=0.01, Maxlter=10, verbose=True):
    paths = []
    for i in range(Maxlter):
        x1 = x0 - learning_rate * grad_func(x0) # -를 해주는 이유: 기울기가 최소인 (안정적인 w값으로)
        if verbose:
            print('{0:003d}:{1:4.3f}:{2:4.2E}'.format(i, x1, func(x1)))
        x0 = x1
        paths.append(x0)
    return(x0, func(x0), paths)

# # 초기값 0, leanring rate을 1.2로 한경우
# xopt, fopt, paths = steepest_descent(f, grad_fx, 0.0, learning_rate = 1.2)

# x = np.linspace(0.5, 2.5, 1000)
# paths = np.array(paths)
# plt.plot(x,f(x))
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('plot of f(x)')

# plt.plot(paths, f(paths), 'o-')
# plt.show()

# plt.plot(f(paths), 'o-')
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('plot of f(x)')

# # 초기값 1, leanring rate을 1로 한경우
# xopt, fopt, paths = steepest_descent(f, grad_fx, 1.0, learning_rate = 1)

# x = np.linspace(0.5, 3.5, 1000)
# paths = np.array(paths)
# plt.plot(x,f(x))
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('plot of f(x)')

# plt.plot(paths, f(paths), 'o-')
# plt.show()

# plt.plot(f(paths), 'o-')
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('plot of f(x)')

# # 초기값 1, leanring rate을 0.001로 한경우
# xopt, fopt, paths = steepest_descent(f, grad_fx, 1.0, learning_rate = 0.001)

# x = np.linspace(0.5, 3.5, 1000)
# paths = np.array(paths)
# plt.plot(x,f(x))
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('plot of f(x)')

# plt.plot(paths, f(paths), 'o-')
# plt.show()

# plt.plot(f(paths), 'o-')
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('plot of f(x)')

# 초기값 1, leanring rate을 0.001로 한경우
xopt, fopt, paths = steepest_descent(f, grad_fx, 3.0, learning_rate = 0.9)

x = np.linspace(0.5, 3.5, 1000)
paths = np.array(paths)
plt.plot(x,f(x))
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')

plt.plot(paths, f(paths), 'o-')
plt.show()

plt.plot(f(paths), 'o-')
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')