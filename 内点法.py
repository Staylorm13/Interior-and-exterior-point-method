from numpy.linalg import norm
from sympy import symbols, flatten
import numpy as np


def f(x):
	f = (x[0] + 1) ** 3 / 3 + x[1]
	return f


def g(x):
	g = [x[0] - 1, x[1]]
	return g


class out_point_method(object):
	def __init__(self, x0):
		self.x0 = x0
	
	def method(self, r, e):
		xk = self.x0
		for i in range(50):
			fx0, gx0 = out_point_method.get_fgx(self, x, xk)
			p = out_point_method.get_Bx(self, gx0)
			grad = out_point_method.get_grad(self, r, xk)
			if r * p <= e or norm(grad) <= e:
				break
			else:
				Jacobi = out_point_method.get_Jacobi(self, r, xk)
				"""
				方程组求解用np.linalg.solve()
				多维数组转化为一维数组用sympy.flatten()
				"""
				xp = flatten(np.linalg.solve(Jacobi, grad))
				xk = np.asarray(xk) - np.asarray(xp)
				r = 0.4 * r
		it = i
		xx = xk
		fmin = out_point_method.get_fgx(self, x, xk)
		print(it, '\n', xx, '\n', fmin[0])
	
	def get_fgx(self, x, xk):
		"""
		多个自变量计算，则需要多次使用subs()
		"""
		fxk = f(x).subs(x[0], xk[0])
		fxk = fxk.subs(x[1], xk[1])
		"""
		若是一个函数组，则采用循环方式进行赋值计算
		"""
		gxk = []
		for j in range(0, len(g(x))):
			gxj = g(x)[j].subs(x[0], xk[0])
			gxj = gxj.subs(x[1], xk[1])
			gxk.append(gxj)
		return fxk, gxk,
	
	def get_Bx(self, gxk):
		Bx = 0
		for k in range(0, len(gxk)):
			Bx = Bx + (1 / gxk[k])
		return Bx
	
	def get_grad(self, r, xk):
		fxk, gxk = out_point_method.get_fgx(self, x, xk)
		Bxk = out_point_method.get_Bx(self, gxk)
		Fk = fxk + r * Bxk
		step = 0.001
		grad = np.zeros((len(xk), 1))
		for k in range(0, len(xk)):
			xkk = xk.copy()
			xkk[k] = xkk[k] + step
			fxkk, gxkk = out_point_method.get_fgx(self, x, xkk)
			p1 = out_point_method.get_Bx(self, gxkk)
			Fkk = fxkk + r * p1
			grad[k] = (Fkk - Fk) / step
		return grad
	
	def get_Jacobi(self, r, xk):
		grad_k = out_point_method.get_grad(self, r, xk)
		m = len(grad_k)
		n = len(xk)
		Jacobi = np.zeros((m, n))
		step = 0.01
		for k in range(0, n):
			tmp = xk.copy()
			tmp[k] = tmp[k] + step
			grad_kk = out_point_method.get_grad(self, r, tmp)
			a = (np.asarray(grad_kk) - np.asarray(grad_k)) / step
			Jacobi[0, k] = a[0]
			Jacobi[1, k] = a[1]
		return Jacobi


if __name__ == '__main__':
	x = [symbols('x1'), symbols('x2')]
	x0 = [2,1]
	r = 10
	e = 0.0001
	opm = out_point_method(x0)
	opm.method(r, e)
