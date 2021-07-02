from numpy.linalg import norm
from sympy import symbols, flatten
import numpy as np


def f(x):
	f = x[0]**2+2*x[1]**2
	return f


def g(x):
	g = [x[0]+x[1]-1]
	return g


def h(x):
	h =0*x[0]+0*x[1]
	return h


class out_point_method(object):
	def __init__(self, x0):
		self.x0 = x0
	
	def method(self, M, e):
		xk = self.x0
		for i in range(50):
			fx0, gx0, hx0 = out_point_method.get_fghx(self, x, xk)
			p = out_point_method.get_px(self, gx0, hx0)
			grad = out_point_method.get_grad(self, M, xk)
			if M * p <= e or norm(grad) <= e:
				break
			else:
				Jacobi = out_point_method.get_Jacobi(self, M, xk)
				"""
				方程组求解用np.linalg.solve()
				多维数组转化为一维数组用sympy.flatten()
				"""
				xp = flatten(np.linalg.solve(Jacobi, grad))
				xk = np.asarray(xk) - np.asarray(xp)
				M = 2 * M
		it = i
		xx = xk
		fmin = out_point_method.get_fghx(self, x, xk)
		print(it, '\n', xx, '\n', fmin[0])
	
	def get_fghx(self, x, xk):
		"""
		多个自变量计算，则需要多次使用subs()
		"""
		fx0 = f(x).subs(x[0], xk[0])
		fx0 = fx0.subs(x[1], xk[1])
		"""
		若是一个函数组，则采用循环方式进行赋值计算
		"""
		gx0 = []
		for j in range(0, len(g(x))):
			gxj = g(x)[j].subs(x[0], xk[0])
			gxj = gxj.subs(x[1], xk[1])
			gx0.append(gxj)
		hx0 = h(x).subs(x[0], xk[0])
		hx0 = hx0.subs(x[1], xk[1])
		return fx0, gx0, hx0
	
	def get_px(self, gx, hx):
		p = 0
		for k in range(0, len(gx)):
			p = p + min(gx[k], 0) ** 2
		return p + hx ** 2
	
	def get_grad(self, M, xk):
		fx0, gx0, hx0 = out_point_method.get_fghx(self, x, xk)
		p = out_point_method.get_px(self, gx0, hx0)
		F0 = fx0 + M * p
		step = 0.001
		grad = np.zeros((len(xk), 1))
		for k in range(0, len(xk)):
			xtmp = xk.copy()
			xtmp[k] = xtmp[k] + step
			fx1, gx1, hx1 = out_point_method.get_fghx(self, x, xtmp)
			p1 = out_point_method.get_px(self, gx1, hx1)
			F1 = fx1 + M * p1
			grad[k] = (F1 - F0) / step
		return grad
	
	def get_Jacobi(self, M, xk):
		grad0 = out_point_method.get_grad(self, M, xk)
		m = len(grad0)
		n = len(xk)
		Jacobi = np.zeros((m, n))
		step = 0.01
		for k in range(0, n):
			tmp = xk.copy()
			tmp[k] = tmp[k] + step
			grad1 = out_point_method.get_grad(self, M, tmp)
			a = (np.asarray(grad1) - np.asarray(grad0)) / step
			Jacobi[0, k] = a[0]
			Jacobi[1, k] = a[1]
		return Jacobi


if __name__ == '__main__':
	x = [symbols('x1'), symbols('x2')]
	x0 = [2, -20]
	M = 3
	e = 0.0001
	opm = out_point_method(x0)
	opm.method(M, e)
