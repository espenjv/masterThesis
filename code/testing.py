import numpy as np
from scipy import misc, ndimage,sparse, signal, io, integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from PIL import Image

f = misc.imread('flow.jpg',mode='L')

[m,n] = f.shape

f_sigma = ndimage.filters.gaussian_filter(f,1.0)

f_sigma = f_sigma.flatten('F')

# g = lambda y,x,grad,f: 1/(1+eta*np.power(grad(y,x,f),2))
#
# delt = 20
#
# print f.shape
#
N = m*n
#
# A = sparse.coo_matrix((N,N))
#
# basis = np.zeros((4,4))
#
u = (np.random.rand(N)-0.5)*10
#
# eps = 0.0001
#
# eta = 100.0
#
# gamma = 0.2
#
#
# # Generating coefficients for bilinear basis functions
# # fn_basis[k] = basis[0,k]xy + basis[1,k]y + basis[2,k]x + basis[3,k]
#
#
# basis = np.zeros((4,4))
# basis[0,0] = 1
# basis[1,0] = -1
# basis[2,0] = -1
# basis[3,0] = 1
#
# basis[0,1] = -1
# basis[1,1] = 1
# basis[2,1] = 0
# basis[3,1] = 0
#
# basis[0,2] = -1
# basis[1,2] = 0
# basis[2,2] = 1
# basis[3,2] = 0
#
# basis[0,3] = 1
# basis[1,3] = 0
# basis[2,3] = 0
# basis[3,3] = 0
#
# basis = 1/4*basis
#
#
#
# gradient_basis = lambda y,x: np.array([basis[0,:]*y + basis[2,:],basis[0,:]*x+basis[1,:]])
# quad_basis = lambda y,x: np.array([basis[0,:]*x*y + basis[1,:]*y + basis[2,:]*x + basis[3,:]])
#
#
# # integrand_M =  lambda y,x,n1,n2,grad,u,g,f: np.divide(delt*g(y,x,grad,f)*((basis[0,n1]*y + basis[2,n1])*(basis[0,n2]*y + basis[2,n2]) + (basis[0,n1]*x+basis[1,n1])*(basis[0,n2]*x+basis[1,n2])) + (basis[0,n1]*x*y + basis[1,n1]*y + basis[2,n1]*x + basis[3,n1]) * (basis[0,n2]*x*y + basis[1,n2]*y + basis[2,n2]*x + basis[3,n2]), grad(y,x,u))
#
#
# # integrand_M =  lambda y,x,n1,n2,grad,u,g,f: np.divide(delt*g(y,x,grad,f)*((basis[0,n1]*y + basis[2,n1])*(basis[0,n2]*y + basis[2,n2]) + (basis[0,n1]*x+basis[1,n1])*(basis[0,n2]*x+basis[1,n2])) + (basis[0,n1]*x*y + basis[1,n1]*y + basis[2,n1]*x + basis[3,n1]) * (basis[0,n2]*x*y + basis[1,n2]*y + basis[2,n2]*x + basis[3,n2]) , np.linalg.norm(np.array([basis[0,:]*y + basis[2,:],basis[0,:]*x+basis[1,:]])*u) + eps)
#
# integrand_M =  lambda y,x,n1,n2,u,f: np.divide(delt*((basis[0,n1]*y + basis[2,n1])*(basis[0,n2]*y + basis[2,n2]) +
# (basis[0,n1]*x+basis[1,n1])*(basis[0,n2]*x+basis[1,n2])) + (basis[0,n1]*x*y + basis[1,n1]*y + basis[2,n1]*x + basis[3,n1]) * (basis[0,n2]*x*y + basis[1,n2]*y + basis[2,n2]*x + basis[3,n2]) ,
# (np.linalg.norm(np.array([basis[0,:]*y + basis[2,:],basis[0,:]*x+basis[1,:]])*u) + eps) * (1+eta*np.power(np.linalg.norm(np.array([basis[0,:]*y + basis[2,:],basis[0,:]*x+basis[1,:]])*f),2)))
#
# #integrand_M =  lambda y,x,n1,n2,grad,u,g,f: np.divide(delt*g(y,x,grad,f)*((gradient_basis(y,x)[:,n1])*(gradient_basis(y,x)[:,n2])+ (quad_basis(y,x)[:,n1]) * (quad_basis(y,x)[:,n2])) , grad(y,x,u))
#
#
# integrand_rhs =  lambda y,x,n1,u,f: (np.divide(np.array([basis[0,:]*x*y + basis[1,:]*y + basis[2,:]*x + basis[3,:]])*u,np.linalg.norm(np.array([basis[0,:]*y + basis[2,:],basis[0,:]*x+basis[1,:]])*u) + eps) +
# np.divide(delt*gamma,1+eta*np.power(np.linalg.norm(np.array([basis[0,:]*y + basis[2,:],basis[0,:]*x+basis[1,:]])*f) + eps,2)))*(basis[0,n1]*x*y + basis[1,n1]*y + basis[2,n1]*x + basis[3,n1])
#
#
#
# rows = []
# columns = []
# data_M = []
#
# rhs_rows = []
# data_rhs = []
#
#
#
# for i in range(m-1):
#     for j in range(n-1):
#
#         indices = [i*j,i*j+1,i*j+m,i*j+m+1]
#         #u_tilde = np.array([u[i*j],u[i*j+1],u[i*j+m],u[i*j+m+1]])
#         #gradient = lambda y,x,w: np.linalg.norm(gradient_basis(y,x)*w) + eps
#         #gradient = lambda y,x,w: np.linalg.norm(np.array([basis[0,:]*y + basis[2,:],basis[0,:]*x+basis[1,:]])*w) + eps
#
#
#
#         u_tilde = u[indices]
#
#         f_sigma_tilde = f_sigma[indices]
#
#         for n1 in range(4):
#
#
#             dat_rhs = 1/4*(integrand_rhs(1/np.sqrt(3),1/np.sqrt(3),n1,u_tilde,f_sigma_tilde) +
#             integrand_rhs(1/np.sqrt(3),-1/np.sqrt(3),n1,u_tilde,f_sigma_tilde) +
#             integrand_rhs(-1/np.sqrt(3),1/np.sqrt(3),n1,u_tilde,f_sigma_tilde) +
#             integrand_rhs(-1/np.sqrt(3),-1/np.sqrt(3),n1,u_tilde,f_sigma_tilde))
#
#             rhs_rows.append(indices[n1])
#             data_rhs.append(dat_rhs)
#             for n2 in range(n1,4):
#
#                 dat_M = 1/4*(integrand_M(1/np.sqrt(3),1/np.sqrt(3),n1,n2,u_tilde,f_sigma_tilde) +
#                 integrand_M(1/np.sqrt(3),-1/np.sqrt(3),n1,n2,u_tilde,f_sigma_tilde) +
#                 integrand_M(-1/np.sqrt(3),1/np.sqrt(3),n1,n2,u_tilde,f_sigma_tilde) +
#                 integrand_M(-1/np.sqrt(3),-1/np.sqrt(3),n1,n2,u_tilde,f_sigma_tilde))
#
#                 rows.append(indices[n1])
#                 columns.append(indices[n2])
#                 data_M.append(dat_M)
#
#                 rows.append(indices[n2])
#                 columns.append(indices[n1])
#                 data_M.append(dat_M)
#
#
#
#
# A = sparse.coo_matrix((data_M,(rows,columns)),shape=(N,N)).tocsr()
# b = sparse.coo_matrix((data_rhs,(rhs_rows,np.ones(N))),shape=(N,1)).tocsr()

print m
print n


def makeLagrangianBasis():

    basis = np.zeros((4,4))
    basis[0,0] = 1
    basis[0,1] = -1
    basis[0,2] = -1
    basis[0,3] = 1

    basis[1,0] = -1
    basis[1,1] = 1
    basis[1,2] = 0
    basis[1,3] = 0

    basis[2,0] = -1
    basis[2,1] = 0
    basis[2,2] = 1
    basis[2,3] = 0

    basis[3,0] = 1
    basis[3,1] = 0
    basis[3,2] = 0
    basis[3,3] = 0

    basis = 1.0/4*basis

    return basis

def evaluate_function(y,x,w,m,n):

    basis = makeLagrangianBasis()

    j = int(x/n)
    i = int(y/m)

    indices = [i*j,i*j+1,i*j+m,i*j+m+1]

    u = 2*(x-j)-1
    v = 2*(y-i)-1


    return np.dot(np.array([basis[:,0]*u*v + basis[:,1]*v + basis[:,2]*u + basis[:,3]]),w[indices])

def evaluate_gradient(y,x,w,m,n):
    basis = makeLagrangianBasis()

    j = int(x/n)
    i = int(y/m)

    indices = [i*j,i*j+1,i*j+m,i*j+m+1]

    u = 2*(x-j)-1
    v = 2*(y-i)-1


    return np.dot(np.array([basis[:,0]*y + basis[:,2],basis[:,0]*x+basis[:,1]]),w[indices])


class JordanCurve:
    def __init__(self,points):
        self.K = len(points)
        self.control_points = np.vstack((points[self.K-1],np.vstack((points,points[0:3]))))

    def evaluate(self,t):
        k = int(t*self.K)
        i = k + 1

        return self.control_points[i+2]*self.a(t) + self.control_points[i+1]*self.b(t) + self.control_points[i]*self.c(t) + self.control_points[i-1]*self.d(t)

    def evaluate_d(self,t):
        k = int(t*self.K)
        i = k + 1

        return self.K*(self.control_points[i+2]*self.a_d(t) + self.control_points[i+1]*self.b_d(t) + self.control_points[i]*self.c_d(t) + self.control_points[i-1]*self.d_d(t))

    def evaluate_dd(self,t):
        k = int(t*self.K)
        i = k + 1

        return self.K*self.K*(self.control_points[i+2]*self.a_dd(t) + self.control_points[i+1]*self.b_dd(t) + self.control_points[i]*self.c_dd(t) + self.control_points[i-1]*self.d_dd(t))




    def a(self,t):
        k = int(t*self.K)
        u = (t - 1.0/self.K*k)*self.K
        return 1.0/6*np.power(u,3)
    def b(self,t):
        k = int(t*self.K)
        u = (t - 1.0/self.K*k)*self.K
        return (-3*np.power(u,3) + 3*np.power(u,2) + 3*u + 1)*1.0/6
    def c(self,t):
        k = int(t*self.K)
        u = (t - 1.0/self.K*k)*self.K
        return 1.0/6*(3*np.power(u,3)-6*np.power(u,2)+4)
    def d(self,t):
        k = int(t*self.K)
        u = (t - 1.0/self.K*k)*self.K
        return 1.0/6*(-np.power(u,3)+3*np.power(u,2)-3*u+1)


    def a_d(self,t):
        k = int(t*self.K)
        u = (t - 1.0/self.K*k)*self.K
        return 1.0/6*3*np.power(u,2)
    def b_d(self,t):
        k = int(t*self.K)
        u = (t - 1.0/self.K*k)*self.K
        return (-3*3*np.power(u,2) + 3*2*u + 3)*1.0/6
    def c_d(self,t):
        k = int(t*self.K)
        u = (t - 1.0/self.K*k)*self.K
        return 1.0/6*(3*3*np.power(u,2)-6*2*u)
    def d_d(self,t):
        k = int(t*self.K)
        u = (t - 1.0/self.K*k)*self.K
        return 1.0/6*(-3*np.power(u,3)+3*2*u-3)

    def a_dd(self,t):
        k = int(t*self.K)
        u = (t - 1.0/self.K*k)*self.K
        return 1.0/6*3*2*u
    def b_dd(self,t):
        k = int(t*self.K)
        u = (t - 1.0/self.K*k)*self.K
        return (-3*3*2*u + 3*2)*1.0/6
    def c_dd(self,t):
        k = int(t*self.K)
        u = (t - 1.0/self.K*k)*self.K
        return 1.0/6*(3*3*2*u-6*2)
    def d_dd(self,t):
        k = int(t*self.K)
        u = (t - 1.0/self.K*k)*self.K
        return 1.0/6*(-3*2*u+3*2)


    def draw_curve(self,m,n,T,rgb):
        dt = 1.0/T
        t = 0
        img = np.ones((m,n,3),dtype=np.uint8)*255
        for i in range(T):
            coord = self.evaluate(t)
            print coord
            img[int(coord[0]),int(coord[1])] = rgb
            t = t + dt
        return img






def evolveControlPoints(closed_curves,delt,u,alpha,beta,f,m,n):
    eps = 0.0001
    # Step length for computation of integral 3.4
    N = len(f)
    dt = 1/N

    T = 10


    u1 = sum(f*(u>=0))/sum(u>=0)
    u2 = sum(f*(u<0))/sum(u>=0)

    def MS_grad(t,C):
        coord = C.evaluate(t)
        grad = evaluate_gradient(coord[1],coord[0],f,m,n)
        normal = grad/np.linalg.norm(grad+ eps)
        C_d_norm = np.linalg.norm(C.evaluate_d(t))
        C_dd = C.evaluate_dd(t)
        return alpha*(np.power(u2-evaluate_function(coord[1],coord[0],u,m,n),2) - np.power(u1-evaluate_function(coord[1],coord[0],u,m,n),2))*C_d_norm*normal + beta*C_d_norm*C_dd

    for j in range(len(closed_curves)):

        C = closed_curves[j]
        coeff_vec = np.zeros((C.K+3,2))
        p = C.control_points

        sc = 1.0/36*1.0/C.K

        A = sparse.diags([np.ones(C.K)*604.0/35*sc,np.ones(C.K-1)*1191.0/140*sc,np.ones(C.K-2)*6.0/7*sc,np.ones(C.K-3)*1.0/140*sc,np.ones(C.K-1)*1191.0/140*sc,np.ones(C.K-2)*6.0/7*sc,np.ones(C.K-3)*1.0/140*sc],[0,1,2,3,-1,-2,-3])
        A = A.tocsr()

        for k in range(C.K):
            print k
            t0 = k*1.0/C.K
            t1 = (k+1)/C.K
            dt = (t1-t0)*1.0/T
            t = t0
            for i in range(T):
                coeff_vec[k-1] = coeff_vec[k] + (1.0/C.K*MS_grad(t,C)*(C.d(t))+ 1.0/C.K*MS_grad(t+dt,C)*(C.d(t+dt)))*1.0/2
                coeff_vec[k] = coeff_vec[k] + (1.0/C.K*MS_grad(t,C)*(C.c(t)) + 1.0/C.K*MS_grad(t+dt,C)*(C.c(t+dt)))*1.0/2
                coeff_vec[k+1] = coeff_vec[k+1] + (1.0/C.K*MS_grad(t,C)*(C.b(t)) + 1.0/C.K*MS_grad(t+dt,C)*(C.b(t+dt)))*1.0/2
                coeff_vec[k+2] = coeff_vec[k+2] + (1.0/C.K*MS_grad(t,C)*(C.a(t)) + 1.0/C.K*MS_grad(t+dt,C)*(C.a(t+dt)))*1.0/2

                t = t + dt
        evolution_vector = coeff_vec[1:C.K+1]
        evolution_vector[C.K-1] = evolution_vector[C.K-1] + coeff_vec[0]
        evolution_vector[0:2] = evolution_vector[0:2] + coeff_vec[-2:]

        print A.shape
        print evolution_vector.shape
        dp = -sparse.linalg.spsolve(A,evolution_vector)
    return 0


control_points = np.array([[100,200],[100,220],[100,240],[100,260],[100,280],[100,300],[120,300],[140,300],[160,300],[180,300],[200,300],[200,280],[200,260],
[200,240],[200,220],[200,200],[180,200],[160,200],[140,200],[120,200]])

curve = JordanCurve(control_points)

red = [255,0,0]

img = curve.draw_curve(m,n,1000,[255,0,0])

img = Image.fromarray(img, 'RGB')

plt.figure()
plt.imshow(img)
plt.axis('off')
plt.show()

#
# curves = [curve]
#
# tes = evolveControlPoints(curves,1.0/50,u,1.0,1.0,f_sigma,m,n)
