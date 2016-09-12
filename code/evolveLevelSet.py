import numpy as np
from scipy import misc, ndimage,sparse
from scipy.sparse.linalg import spsolve
from PIL import Image
import matplotlib.pyplot as plt




def makeLagrangianBasis():

    # Returns a 4x4 matrix with the 4 basis coefficients of the 4 bilinear basis
    # functions on the square [-1,1]x[-1,1]
    # The coefficients are ordered as follows:
    # b_n(x,y) = basis[n,0]xy + basis[n,1]y + basis[n,2]x + basis[n,3]

    basis = np.zeros((4,4))
    basis[0,0] = 1.0
    basis[0,1] = -1.0
    basis[0,2] = -1.0
    basis[0,3] = 1.0

    basis[1,0] = -1.0
    basis[1,1] = 1.0
    basis[1,2] = 0.0
    basis[1,3] = 0.0

    basis[2,0] = -1.0
    basis[2,1] = 0.0
    basis[2,2] = 1.0
    basis[2,3] = 0.0

    basis[3,0] = 1.0
    basis[3,1] = 0.0
    basis[3,2] = 0.0
    basis[3,3] = 0.0

    basis = 1.0/4*basis

    return basis



def assemble_system(eps,eta,gamma,f_sigma,delt,u,m,n):
    # Assembles the system for evolving the level set curve
    # eps is a regularization parameter
    # eta is sensitivity parameter for the edge detector function
    # gamma is the baloon force
    # f_sigma is the convolved image
    # delt is the time step
    # u are the discrete level set function values of the current time step
    # m x n is the image size

    N = m*n


    basis = makeLagrangianBasis()


    evaluate_gradient_norm = lambda y,x,w: np.linalg.norm(np.dot(2.0*np.array([basis[:,0]*y + basis[:,2],basis[:,0]*x+basis[:,1]]),w)) + eps
    #evaluate_function = lambda y,x,w: np.array([basis[:,0]*x*y + basis[:,1]*y + basis[:,2]*x + basis[:,3]])*w
    g = lambda y,x,f: np.divide(1.0,(1.0+eta*np.power(evaluate_gradient_norm(y,x,f),2)))

    integrand_M =  lambda y,x,n1,n2,u,f: np.divide(delt*g(y,x,f)*(4.0*(basis[n1,0]*y + basis[n1,2])*(basis[n2,0]*y + basis[n2,2]) + 4.0*(basis[n1,0]*x+basis[n1,1])*(basis[n2,0]*x+basis[n2,1])) + (basis[n1,0]*x*y + basis[n1,1]*y + basis[n1,2]*x + basis[n1,3]) * (basis[n2,0]*x*y + basis[n2,1]*y + basis[n2,2]*x + basis[n2,3]), evaluate_gradient_norm(y,x,u))

    integrand_rhs =  lambda y,x,n1,u,f: (np.divide(np.dot(np.array([basis[:,0]*x*y + basis[:,1]*y + basis[:,2]*x + basis[:,3]]),u),evaluate_gradient_norm(y,x,u)) + delt*gamma*g(y,x,f))*(basis[n1,0]*x*y + basis[n1,1]*y + basis[n1,2]*x + basis[n1,3])



    rows = []
    columns = []
    data_M = []

    rhs_rows = []
    data_rhs = np.zeros((N,1))


    NW = -1



    for j in range(n-1):
        NW += 1
        SW = NW + 1
        NE = NW + m
        SE = NE + 1
        for i in range(m-1):

            print i
            print j

            indices = [NW,SW,NE,SE]
            #indices = [i*j,i*j+1,i*j+m,i*j+m+1]
            #u_tilde = np.array([u[i*j],u[i*j+1],u[i*j+m],u[i*j+m+1]])
            #gradient = lambda y,x,w: np.linalg.norm(gradient_basis(y,x)*w) + eps


            u_tilde = u[indices]

            f_sigma_tilde = f_sigma[indices]

            for n1 in range(4):

                dat_rhs = 1.0/4*(integrand_rhs(1.0/np.sqrt(3),1.0/np.sqrt(3),n1,u_tilde,f_sigma_tilde) +
                integrand_rhs(1.0/np.sqrt(3),-1.0/np.sqrt(3),n1,u_tilde,f_sigma_tilde) +
                integrand_rhs(-1.0/np.sqrt(3),1.0/np.sqrt(3),n1,u_tilde,f_sigma_tilde) +
                integrand_rhs(-1.0/np.sqrt(3),-1.0/np.sqrt(3),n1,u_tilde,f_sigma_tilde))

                data_rhs[indices[n1]] += dat_rhs


                for n2 in range(n1,4):

                    dat_M = 1.0/4*(integrand_M(1.0/np.sqrt(3),1.0/np.sqrt(3),n1,n2,u_tilde,f_sigma_tilde) +
                    integrand_M(1.0/np.sqrt(3),-1.0/np.sqrt(3),n1,n2,u_tilde,f_sigma_tilde) +
                    integrand_M(-1.0/np.sqrt(3),1.0/np.sqrt(3),n1,n2,u_tilde,f_sigma_tilde) +
                    integrand_M(-1.0/np.sqrt(3),-1.0/np.sqrt(3),n1,n2,u_tilde,f_sigma_tilde))

                    # print integrand_M(1/np.sqrt(3),1/np.sqrt(3),n1,n2,u_tilde,f_sigma_tilde)
                    # print integrand_M(1/np.sqrt(3),-1/np.sqrt(3),n1,n2,u_tilde,f_sigma_tilde)
                    # print integrand_M(-1/np.sqrt(3),1/np.sqrt(3),n1,n2,u_tilde,f_sigma_tilde)
                    # print integrand_M(-1/np.sqrt(3),-1/np.sqrt(3),n1,n2,u_tilde,f_sigma_tilde)
                    #
                    #
                    # print dat_M
                    #
                    # raw_input("Enter")

                    rows.append(indices[n1])
                    columns.append(indices[n2])
                    data_M.append(dat_M)

                    rows.append(indices[n2])
                    columns.append(indices[n1])
                    data_M.append(dat_M)
            NW += 1
            SW = NW + 1
            NE = NW + m
            SE = NE + 1


    A = sparse.coo_matrix((data_M,(rows,columns)),shape=(N,N)).tocsr()
    # print A


    return A,data_rhs



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


    def set_points(self,points):
        self.K = len(points)
        self.control_points = np.vstack((points[self.K-1],np.vstack((points,points[0:3]))))

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

    def draw_curve(self,img,T,rgb):
        dt = 1.0/T
        t = 0
        for i in range(T):
            coord = self.evaluate(t)
            img[int(coord[0]),int(coord[1])] = rgb
            t = t + dt
        return img



def evaluate_function(y,x,w,m,n):
    # Evaluates a function in point (x,y) using bilinear basis functions
    # Discrete function values are given in the vector w of length m x n
    # m x n is the size of the domain (image)

    basis = makeLagrangianBasis()

    j = int(x/n)
    i = int(y/m)

    indices = [i*j,i*j+1,i*j+m,i*j+m+1]

    q = 2*(x-j)-1
    z = 2*(y-i)-1


    return np.dot(np.array([basis[:,0]*q*z + basis[:,1]*z + basis[:,2]*q + basis[:,3]]),w[indices])


def evaluate_gradient(y,x,w,m,n):
    # Evaluates the gradient of a function in point (x,y) using bilinear basis
    # w are the discrete function values
    # m x n is the size of the image

    basis = makeLagrangianBasis()

    j = int(x/n)
    i = int(y/m)

    indices = [i*j,i*j+1,i*j+m,i*j+m+1]

    q = 2*(x-j)-1
    z = 2*(y-i)-1

    gradient = np.zeros((2))

    for k in range(4):
        gradient += 2*w[indices[k]]*np.array([basis[k,0]*z + basis[k,2],basis[k,0]*q+basis[k,1]])

    return gradient


def evolveControlPoints(closed_curves,delt,u,alpha,beta,f,m,n):
    # Method for evolving Control points


    eps = 0.0001
    # Step length for computation of integral 3.4
    N = len(f)
    dt = 1.0/N

    T = 50


    # u1 = sum(f*(u>=0))/sum(u>=0)
    # u2 = sum(f*(u<0))/sum(u<0)

    u1 = 44
    u2 = 255

    print u1
    print u2


    def MS_grad(t,C):
        # Gradient of the Mumford-Shah energy functional
        # t is the current time
        # C is a JordanCurve

        coord = C.evaluate(t)
        grad = evaluate_gradient(coord[1],coord[0],u,m,n)
        # print grad
        normal = grad/np.linalg.norm(grad + eps)
        C_d_norm = np.linalg.norm(C.evaluate_d(t))
        C_dd = C.evaluate_dd(t)
        # # print evaluate_function(coord[1],coord[0],u,m,n)
        # print C.evaluate_d(t)
        # print C_d_norm
        print C_dd
        # print normal
        # print evaluate_function(coord[1],coord[0],f,m,n)
        # print np.power(u2-evaluate_function(coord[1],coord[0],f,m,n),2)
        # print np.power(u1-evaluate_function(coord[1],coord[0],f,m,n),2)
        # print np.power(u2-evaluate_function(coord[1],coord[0],f,m,n),2) - np.power(u1-evaluate_function(coord[1],coord[0],f,m,n),2)
        # print beta*C_d_norm*C_dd
        return alpha*(np.power(u2-evaluate_function(coord[1],coord[0],f,m,n),2) - np.power(u1-evaluate_function(coord[1],coord[0],f,m,n),2))*C_d_norm*normal + beta*C_d_norm*C_dd

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
            dt = 1/(C.K*T)
            t = t0
            for i in range(T):
                # print MS_grad(t,C)
                # raw_input("Press Enter")
                print (1.0/C.K)*MS_grad(t,C)
                print dt*(1.0/C.K*MS_grad(t,C)*(C.d(t))+ 1.0/C.K*MS_grad(t+dt,C)*(C.d(t+dt)))*1.0/2
                raw_input()
                coeff_vec[k-1] = coeff_vec[k-1] + dt*(1.0/C.K*MS_grad(t,C)*(C.d(t))+ 1.0/C.K*MS_grad(t+dt,C)*(C.d(t+dt)))*1.0/2
                coeff_vec[k] = coeff_vec[k] + dt*(1.0/C.K*MS_grad(t,C)*(C.c(t)) + 1.0/C.K*MS_grad(t+dt,C)*(C.c(t+dt)))*1.0/2
                coeff_vec[k+1] = coeff_vec[k+1] + dt*(1.0/C.K*MS_grad(t,C)*(C.b(t)) + 1.0/C.K*MS_grad(t+dt,C)*(C.b(t+dt)))*1.0/2
                coeff_vec[k+2] = coeff_vec[k+2] + dt*(1.0/C.K*MS_grad(t,C)*(C.a(t)) + 1.0/C.K*MS_grad(t+dt,C)*(C.a(t+dt)))*1.0/2

                t = t + dt
        evolution_vector = coeff_vec[1:C.K+1]
        evolution_vector[C.K-1] = evolution_vector[C.K-1] + coeff_vec[0]
        evolution_vector[0:2] = evolution_vector[0:2] + coeff_vec[-2:]

        print evolution_vector

        print A.shape
        print evolution_vector.shape
        dp = -sparse.linalg.spsolve(A,evolution_vector)
        print dp
        print p.shape
        raw_input()
        new_points = p[1:C.K+1] + delt*dp
        C.set_points(new_points)

    return closed_curves


def drawLevelSet(u,f,m,n,rgb):
    u = np.reshape(u,(n,m)).T
    img = np.ones((m,n,3),'uint8')
    for i in range(m):
        for j in range(n):
            if u[i,j] < 0.001:
                img[i,j] = rgb
            else:
                img[i,j] = f[i,j]
    return img



f = misc.imread('circle_color.jpg',mode='L')

f_color = misc.imread('circle_color.jpg')



[m,n] = f.shape

#
# H = np.array([[1 ,2, 3,5],[4 ,5 ,6,8],[7, 8 ,9,10]])

# print H.shape
#
# print H
# H = H.flatten('F')
#
# print H
#
# f_color = f_color[0:int(m/5),0:int(n/5),:]


# plt.figure()
# plt.imshow(f_color)
# plt.axis('off')
# plt.show()


# m= int(m/5)
# n = int(n/5)

f_sigma = ndimage.filters.gaussian_filter(f,0.5)

# plt.figure()
# plt.imshow(f_sigma,'gray')
# plt.axis('off')
# plt.show()

f_sigma = f_sigma.flatten('F')



delt = 20.0
alpha = 10.0
beta = 0.1
eps = 0.1
eta = 5.0
gamma = 1.0
# u1 = 44
# u2 = 255



u = np.zeros((m,n))
# u[20:m-20,20:n-20] = np.ones((m-40,n-40))

for i in range(m):
    for j in range(n):
        u[i,j] = (-np.sqrt(np.power(i-m*1.0/2,2) + np.power(j-n*1.0/2,2)) + 10)



# zero_set = u==0
# #
# zero_set = np.reshape(zero_set,(m,n))
# #
# plt.figure()
# plt.imshow(zero_set,'gray')
# plt.axis('off')
# plt.show()

# plt.figure()
# plt.imshow(zero_set,'gray')
# plt.axis('off')
# plt.show()

u = u.flatten('F')


img = drawLevelSet(u,f_color,m,n,[0, 255, 0])


plt.figure()
plt.imshow(img)
plt.axis('off')
plt.show()

control_points = np.array([[0,0],[0,int(n*1.0/5)],[0,int(n*2.0/5)],[0,int(n*3.0/5)],[0,int(n*4.0/5)],[0,int(n*5.0/5)-1],[int(m*1.0/5),n-1],[int(m*2.0/5),n-1],
[int(m*3.0/5),n-1],[int(m*4.0/5),n-1],[m-1,n-1],
[m-1,int(n*4.0/5)],[m-1,int(n*3.0/5)],[m-1,int(n*2.0/5)],[m-1,int(n*1.0/5)],[m-1,0],
[int(m*4.0/5),0],[int(m*3.0/5),0],[int(m*2.0/5),0],[int(m*1.0/5),0]])


curve = JordanCurve(control_points)

curves = [curve]

red = np.array([255,0,0])

T = 500

img_curve = np.copy(f_color)
for curve in curves:
    img_curve= curve.draw_curve(img_curve,T,red)

img_curve = Image.fromarray(img_curve, 'RGB')


plt.figure()
plt.imshow(img_curve)
plt.axis('off')
plt.show()


curves = evolveControlPoints(curves,1.0/50,u,alpha,beta,f_sigma,m,n)

#
# img = curve.draw_curve(f_color,1000,red)
#
# img = Image.fromarray(img, 'RGB')
#
# plt.figure()
# plt.imshow(img)
# plt.axis('off')
# plt.show()
#
# curves = evolveControlPoints(curves,1.0/50,u,alpha,beta,f_sigma,m,n)
#
#
#
# img = curve.draw_curve(f_color,1000,red)
#
# img = Image.fromarray(img, 'RGB')
#
# plt.figure()
# plt.imshow(img)
# plt.axis('off')
# plt.show()

A,b = assemble_system(eps,eta,gamma,f_sigma,delt,u,m,n)
u_new = spsolve(A,b)

# Evolving the curve
curves = evolveControlPoints(curves,1.0/50,u_new,alpha,beta,f_sigma,m,n)

# img = np.ones((m,n,3),dtype=np.uint8)*255
img_curve = np.copy(f_color)
for curve in curves:
    img_curve= curve.draw_curve(img_curve,T,red)

img_curve = Image.fromarray(img_curve, 'RGB')


plt.figure()
plt.imshow(img_curve)
plt.axis('off')
plt.show()


while np.max(np.absolute(u_new-u)) > 0.1:

    print np.max(np.absolute(u_new-u))

    u = u_new

    img = drawLevelSet(u,f_color,m,n,[0, 255, 0])



    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Evolving the level set
    A,b = assemble_system(eps,eta,gamma,f_sigma,delt,u,m,n)
    u_new = spsolve(A,b)



    # Evolving the curve
    curves = evolveControlPoints(curves,1.0/50,u_new,alpha,beta,f_sigma,m,n)

    # img = np.ones((m,n,3),dtype=np.uint8)*255
    img_curve = np.copy(f_color)
    for curve in curves:
        img_curve= curve.draw_curve(img_curve,T,red)

    img_curve = Image.fromarray(img_curve, 'RGB')


    plt.figure()
    plt.imshow(img_curve)
    plt.axis('off')
    plt.show()


# zero_set = u<gamma
#
# zero_set = np.reshape(zero_set,(m,n))
#
# plt.figure()
# plt.imshow(zero_set,'gray')
# plt.axis('off')
# plt.show()


    # # Evolving the curve
    # curves = evolveControlPoints(curves,1.0/50,u,alpha,beta,f_sigma,m,n)
    #
    # # img = np.ones((m,n,3),dtype=np.uint8)*255
    # for curve in curves:
    #     f_curve = curve.draw_curve(f_color,1000,red)
    #
    #     f_curve = Image.fromarray(f_curve, 'RGB')
    #
    #
    # plt.figure()
    # plt.imshow(f)
    # plt.axis('off')
    # plt.show()
