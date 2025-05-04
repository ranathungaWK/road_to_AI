import numpy as np
from numpy.lib import stride_tricks
# import mediapipe as mp
# import cvzone as cvz
# import cv2 as cv
# import datetime

# vector = np.arange(10,100,1)
# vector = vector[::-1]

# vector = np.arange(9).reshape(3, 3)
# print(vector)


# non_zero = np.nonzero([1,2,0,0,5,0,6,2]) # find indices of non zero elements
# print(non_zero)

# identity_matrix = np.identity(5)
# identity_matrix_2 = np.eye(5)
# print(identity_matrix_2)

# random_array = np.random.random((2,3))
# random_array_2 = np.random.permutation([1,2,3])
# print(random_array_2)

# random_array_3 = np.random.randint (0,10,(2,3))
# max_int , min_int = random_array_3.max(), random_array_3.min()
# print(random_array_3,"\n")
# print(max_int, min_int)

# random_array_4 = np.random.randint(0,100,30)
# # mean_ = np.mean(random_array_4)
# mean_ = random_array_4.mean()
# print(random_array_4)
# print(mean_)

# all_ones = np.ones((10,10))
# for i in range(1,9):
#     for j in range(1,9):
#         all_ones[i][j] = 0
# print(all_ones,"\n")

# all_ones = np.ones((10,10))
# all_ones[0:10,0] =0
# all_ones[0:10,9] =0
# all_ones[0,0:10] =0
# all_ones[9,0:10] =0
#
# print(all_ones)

# array_ = np.random.randint(0,10,(10,10))
# padded_array = np.pad(array_, pad_width=1, mode='constant', constant_values=0)
# print(padded_array)

# array__ = np.random.randint(0,10, 10)
# print(array__)
# padded_array__ = np.pad(array__, pad_width=2, mode='reflect')
# print(padded_array__)
# padded_array___ = np.pad(array__, pad_width=2, mode='symmetric')
# print(padded_array___)

# array_3 = np.array([1,2,1])
# padded_array_3 = np.pad(array_3, 1, 'constant')
# padded_array_4 = np.pad(array_3, 3, 'reflect')
# print(padded_array_4)

#print(1*np.nan)
# result = np.exp(1000)  # Exponential of a large number
# print(result)

# array_1 = np.array([1,2,3,4])
# array_1_matrix = np.diag(array_1)
# array_1_matrix_2 = np.diag(array_1_matrix)
# matrix_1 = np.diag(array_1, k=1)
# matrix_2 = np.diag(np.arange(1,5), k=1)
# matrix_2 = np.diag(np.arange(1,5), k=-1)
# print(matrix_2)

# matrix = np.zeros((8,8))
# for i in range(8):
#     for j in range(8):
#         if i%2==0 and j%2==1:
#             matrix[i][j] = 1
#         elif i%2==1 and j%2==0:
#             matrix[i][j] = 1
# print(matrix)

# matrix = np.ones((8,8))
# matrix[::2,::2]=0
# matrix[1::2,1::2] = 0
# print(matrix)

# array_3d = np.arange(336).reshape(6,7,8)
# print(array_3d)
# print(np.unravel_index(335,(6,7,8)))

# array_2 = np.array([1, 2, 3])
# tiled_array = np.tile(array_2,(4,5))
# print(tiled_array)
# tiled_array = np.tile([[1,0],[0,1]] , (4,4))
# print(tiled_array)

# Min-Max Normalization

# matrix = np.random.randint(0,10,(5,5))
# print(matrix)
# #---min max normalization---#
# matrix_min = matrix.min()
# matrix_max = matrix.max()
# print(matrix_min)
# print("\n")
# print(matrix_max)
# print("\n")
# print(matrix_max-matrix_min)
# print("\n")
# normalized_matrix = (matrix - matrix_min) / (matrix_max - matrix_min)
# print("Min-Max Normalized Matrix:\n", normalized_matrix,"\n")

#---z-score normalization---#
# Z-Score Normalization
# matrix_mean = matrix.mean()
# matrix_std = matrix.std()
# print(matrix_mean)
# print("\n")
# print(matrix_std)
# print("\n")
# normalized_matrix_z = (matrix - matrix_mean) / matrix_std
# print("Z-Score Normalized Matrix:\n", normalized_matrix_z)

# matrix = np.random.randint(0,10,(5,5))
# normalized_matrix = (matrix - matrix.mean())/matrix.std()
# print(normalized_matrix)

# dtype_int32 = np.dtype('int32')
# print(dtype_int32)
# array1 = np.array([1,2,3,4,5,6,7,8,9,10])
# print(array1.dtype)


# Define a structured dtype
# dtype_struct = np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f4')])
#
# # Create an array with the structured dtype
# array = np.array([('Alice', 25, 55.5), ('Bob', 30, 85.7)], dtype=dtype_struct)
# print(array)
# print(sys.getsizeof(array))
# print(sys.getsizeof(array[0]))
# print(sys.getsizeof(array[1][2]))

# dtype_struct_2 = np.dtype([("r" , np.ubyte , 1) , ("g" , np.ubyte , 1) , ("b" , np.ubyte , 1) , ("a" , np.ubyte , 1) ])
# color = np.array([(255, 0, 0 , 0 ) , (0 ,0 ,255 ,0)], dtype=dtype_struct_2)
# print(color['r'] , "\n")
# print(color['g'] , "\n")
# print(color.shape)

# array_5 = np.random.randint(0,10,(3,3))
# array_6 = np.random.randint(0,10,(3,3))
# print(array_5)
# print("\n")
# print(array_6)
# print("\n")
# multi = array_5 @ array_6
# print(multi)
# print("\n")
# multi_2 = np.multiply(array_5,array_6)
# print(multi_2)
# print("\n")
# multi_3 = np.dot(array_5,array_6)
# print(multi_3)

# array_3 = np.arange(1,11)
# print(array_3)
# array_4 = np.array(array_3[(3<array_3) & (array_3<9)])
# print(array_4)


# # print(sum(range(5),-1))
# from numpy import *
# # print(sum(range(5),-1))
# array=np.random.randint(0,10,(5,5))
# print(array)
# print(sum(array))
# print(sum(array,axis=0))
# print(sum(array,axis=1))

# z = np.array([1,2,3,4,5])
# x = np.array([5,4,3,2,1])
# print(z**z)
# print(z*z)
# print(z << 2)
# print(z >> 2)
# print(z*1j)
# print(z/1/1)
# print(z/1)
# print(z<x)
# print(x<-z)

# print(np.array(0))
# print(np.array(0) / np.array(0))
# print(np.array(0) // np.array(0))
# print(np.array([np.nan]).astype(int).astype(float))

# Z = np.random.uniform(-10,+10,10)
# print(Z)
# random_number_10_20 = np.random.uniform(10, 20,(5,5))
# print(random_number_10_20)
# print (np.copysign(np.ceil(np.abs(Z)), Z))


# Define the arrays
# magnitudes = np.array([1, -2, 3, -4])
# signs = np.array([-1, 1, -1, 1])
#
# # Apply np.copysign
# result = np.copysign(magnitudes, signs)
#
# print(result)

# z1=np.random.randint(0,10,10)
# z2=np.random.randint(0,10,10)
# print(z1,"\n\n" , z2 , "\n\n")
# print(np.intersect1d(z1,z2))


# Set error handling to raise an exception for division by zero
# def error_fun(a,b):
#
#     return print("Error",a,b)
#
# np.seterrcall(error_fun)
# np.seterr(divide='call')
#
# # Array with values
# a = np.array([1.0, 0.0, -1.0])
# b = np.array([0.0, 0.0, 0.0])
# result = np.divide(a, b)
# try:
#     result = np.divide(a, b)
# except FloatingPointError as e:
#     print("FloatingPointError:", e)
# current_settings = np.geterr()
# print(current_settings)
# np.seterr(divide='call')
# current_settings = np.geterr()
# print(current_settings)

# np.sqrt(-1) == np.emath.sqrt(-1)

# yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
# today     = np.datetime64('today', 'D')
# tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
# y = np.datetime64('2024-11-20', 'D')
# now = datetime.datetime.now()
# print(yesterday , today , tomorrow)
# print(y)
# print(now)


# z=np.arange('2024-10' , '2024-11' , dtype='datetime64[D]')
# print(z)

# A = np.ones(3)
# B = np.ones(3)*2
# np.add(A, B, out=B)
# np.divide(A, 2, out=A)
# np.negative(A, out=A)
# np.multiply(A, B, out=A)
# print(A)

# z = np.random.uniform(0,100,10)
# print(z)
# print(z%1)
# print(z-z%1)
# print(np.floor(z))
# print(np.ceil(z)-1)
# print(z.astype(int))
# print(np.trunc(z))

# z = np.zeros((5,5))
# z += np.arange(5)
# print(z)

# def generator_obj():
#     for i in range(5):
#         yield i*i
#
# my_num =generator_obj()
#
# print(next(my_num))
# print(next(my_num))
# print(next(my_num))
# print(next(my_num))
# print(next(my_num))
#
#
# array_w =np.fromiter(my_num,dtype=int,count=5)
# print(array_w)


# z = np.linspace(0,1,11,endpoint=False)
# print(z[1:] , z[:10])
# print(z[:10])

# list_1 = [1,3,2,4,5,8,7,6]
# list_1.sort()
# print(list_1)

# Z = np.array([[1, 2], [3, 4]])
# sum = np.sum(Z, axis=0) column wise
# print(sum)
# sum = np.sum(Z, axis=1) row wise
# print(sum)
# sum = np.add.reduce(Z)
# print(sum)
# print(Z)

# z_1 = np.random.random(10)
# print(z_1)
# z_2 = np.random.random(10)
# print(z_2)


# equal = np.allclose(z_1, z_2, rtol=1e-05, atol=1e-08)
# print(equal)
# equal = np.array_equal(z_1, z_2)
# print(equal)


# z_3 = np.random.randint(10)
# z_4 = np.random.randint(10)
# equal = np.array_equal(z_4, z_3)
# print(equal)


# z = np.arange(10)
# z.flags.writeable = False
# z.flags.writeable = True
# z[0] = 90
# print(z)


# Z = np.random.random((10,2))
# Z[0,:]=1
# X,Y = Z[:,0], Z[:,1]
# print(Z)
# print(X)
# print(Y)
# # # R = np.sqrt(X**2+Y**2)
# T = np.arctan2(Y,X)
# t2 = np.arctan(X)
# # # print(R)
# print(T)
# print(t2)


# Z = np.random.random(10)
# print(Z)
# print(Z.argmax())
# Z[Z.argmax()] = 0
# print(Z)

# Z_ = np.zeros((5,5))
# Z = np.zeros((5,5), [('x',float),('y',float)])
# print(Z , "\n")
#
# Z['x'] , Z['y'] = np.meshgrid(np.linspace(0,1,5),np.linspace(0,1,5))
# print(Z , "\n")
# print(Z['x'] ,"\n\n" , Z['y'] )


# A = np.array([1,7,3])
# B = np.array([4,0,6])
# C = np.empty((3,3))
#
# for i in range(3):
#     for j in range(3):
#         C[i][j] = 1/(A[i]-B[j])
#
# print(C , "\n")
#
# cauchy_matrix = 1.0/np.subtract.outer(A,B)
# print(cauchy_matrix)


# for dtype in [np.int8, np.int32, np.int64]:
#    print(np.iinfo(dtype).min)
#    print(np.iinfo(dtype).max , "\n")
# for dtype in [np.float32, np.float64]:
#    print(np.finfo(dtype).min)
#    print(np.finfo(dtype).max)
#    print(np.finfo(dtype).eps)

# z = np.array([1,2,3])
# y = np.array([4,6,7])
# d = np.abs(z-y)
# p = d[d.argmin()]
# print(p)

# z = np.zeros(10,dtype=[('position',[('x',float,1),('y',float,1)]),('color',[('r',int,1),('g',int,1),('b',int,1)])])
# print(z)
# z[0]['position']['x']= 3.0
# z[0]['position']['y']= 2.0
# z[0]['color']['r']= 255
# z[0]['color']['g']= 255
# z[0]['color']['b']= 255

# print(z)



# coodinates = np.zeros(5,dtype=[('position',[('x',int,1),('y',int,1)])])
# coodinates['position']['x'] = np.random.randint(0,100,5)
# coodinates['position']['y'] = np.random.randint(0,100,5)
# for pos in coodinates['position']:
#     print(pos)
# print(coodinates)


# z = np.random.randint(2,5,(10,2))
# X,Y = np.atleast_2d(z[:,0], z[:,1])
# # print(X)
# d = np.sqrt((X - X.T) ** 2 + (Y - Y.T) ** 2)
# print(d)


# Z = np.arange(10, dtype=np.float32)
# Z = Z.astype(np.int32, copy=True)
# print(Z)


# from io import StringIO
#
# # Fake file
# s = StringIO("""1, 2, 3, 4, 5\n
#                 6,  ,  , 7, 8\n
#                  ,  , 9,10,11\n""")
# Z = np.genfromtxt(s, delimiter=",", dtype=int)
# print(Z)


# Z = np.arange(9).reshape(3,3)
# print(Z,'\n')
# for index, value in np.ndenumerate(Z):
#     print(index, value)
# print("\n")
# for index in np.ndindex(Z.shape):
#     print(index, Z[index])


# X , Y = np.meshgrid( np.linspace(-1,1,100), np.linspace(-1,1,100) )
# D = np.sqrt(X ** 2 + Y ** 2)
# Gaussian = np.exp(-(D-0)**2/2*1*1)
# print(Gaussian)


# n = 10
# p = 3
# Z = np.zeros((n,n))
# np.put(Z, np.random.choice(range(n*n),50, replace=True),np.random.choice(range(100)))
# print(Z)


# Z = np.random.random((5,5))
# # print(Z)
# M = Z -Z.mean(axis=1,keepdims=True)
# print(M)


# Z = np.random.randint(0,10,(3,3))
# print(Z)
# print(Z[Z[:,2].argsort()])


# Z = np.random.randint(0,1,(3,10))
# print(Z)
# print((~Z.any(axis=1)).any())


# Z = np.random.randint(0,10,(5,5))
# print(Z)
# z=5
# m = Z.flat[np.abs(Z-z).argmin()]
# print(m)


# A = np.arange(3).reshape(3,1)
# B = np.arange(3).reshape(1,3)
# print(A,"\n",B)
# Z = np.nditer([A,B,None])
# for x,y,z in Z:
#     z[...] = x+y
# print(Z.operands[2])


# z = np.random.randint(0,10,20)
# print(z)
# I= np.bincount(z,minlength=20)
# print(I)


# z = [1,2,1,4,5,2,3,4,5,6]
# i = [1,6,3,4,5,6,3,2,3,2]
# k = np.bincount(z,i)
# print(k)


# w,h = 16 ,16
# z = np.random.randint(0,2,(w,h,3)).astype(np.ubyte)
# F = z[...,0]*(256*256) + z[...,1]*(256) + z[...,2]
# print(F)
# n = len(np.unique(F))
# print(n)


# z = np.random.randint(0,10,(2,3,3,3))
# print(z)
# A = np.random.randint(0,10,(3,4,3,4))
# # solution by passing a tuple of axes (introduced in numpy 1.7.0)
# sum = A.sum(axis=(-2,-1))
# print(sum)
# # solution by flattening the last two dimensions into one
# # (useful for functions that don't accept tuples for axis argument)
# sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
# print(sum)


# z = np.random.uniform(0,1,100)
# # print(z)
# p = np.random.randint(0,10,100)
# z_sums = np.bincount(p,weights=z)
# z_count = np.bincount(p)
# z_means = z_sums/z_count
# print(z_means)


# A = np.random.randint(0,100,(5,5))
# B = np.random.randint(0,100,(5,5))
#
# final = np.diag(np.dot(A,B))
# final_1 = np.sum(A*B.T,axis=1)
# final_2 = np.einsum("ij,ji->i",A,B)
# print(final)
# print(final_1)
# print(final_2)


# z = np.array([1,2,3,4,5,6])
# # nz = 3
# # z0 = np.zeros(len(z)+(len(z)-1)*nz)
# # z0[0:len(z0)+1:4] = z
# # print(z0)


# A = np.ones((5,5,3))
# B = 2*np.ones((5,5))
# # print(A * B[:,:,None])
# # print(A*B)
# print(B[:,:,None])
# print("\n" , B[:,None,:])
# print("\n" , B[None,:,:])


# z = np.random.randint(0,25,(5,5))
# print(z)
# print("\n")
# z[[0,4]] = z[[1,0]]
# print(z)


# z = np.random .randint(0,100,(10,3))
# F = np.roll(z.repeat(2,axis=1),1,axis=1)
# F = F.reshape(len(F)*3,2)
# G = (F.view( dtype=[("p0",int),("p1",int)]))
# print(len(G))
# G =np.unique(G)
# print(len(G))


# c = [1,1,2,3,4,4,4,5]
# d = np.bincount(c)
# A = np.repeat(np.arange(len(d)), d)
# print(A)


# array = [1,2,3,4,5,6]
# window = 3
# avg_array = []
# for i  in range(len(array)):
#     window_ = array[i:i+window-1]
#     avg = sum(window_)/(window)
#     avg_array.append(avg)
#
# print(avg_array)


# def moving_avg(arr,window_size):
#     cumsum = np.cumsum(arr)
#     cumsum = np.insert(cumsum,0,0)
#     averages = (cumsum[window_size:] - cumsum[:-window_size])/window_size
#     return averages
#
# print(moving_avg([1,2,3,4,5,6,7,8,9,10],3))


from numpy.lib.stride_tricks import as_strided
# def rolling(array , window):
#     shape = (array.size - window +1 , window)
#     strides = (array.strides[0],array.strides[0])
#     return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
#
# z = rolling(np.array([1,2,3,4,5,6,7,8,9]), 3)
# print(z)


# def distance(array_p0 , array_p1 , array_p):
#     p0_p1 = array_p1 - array_p0
#     L = (p0_p1**2).sum(axis=1)
#     U = ((array_p0[:,0] - array_p[:,0])*p0_p1[:,0] + (array_p0[:,1]- array_p[:,1])*p0_p1[:,0]) / L
#     # return U
#     U = U.reshape(len(U),1)
#     # return U
#     D = array_p0 + U*p0_p1 - array_p
#     return np.sqrt((D**2).sum(axis=1))
#
# # for one P point ----
# # P0 = np.random.uniform(-10,10,(10,2))
# # P1 = np.random.uniform(-10,10,(10,2))
# # P2 = np.random.uniform(-10,10,(1,2))
# #
# # print(distance(P0,P1,P2))
#
# # for number of P points -------
# P0 = np.random.uniform(-10,10,(10,2))
# P1 = np.random.uniform(-10,10,(10,2))
# P2 = np.random.uniform(-10,10,(10,2))
#
# # print(P2)
#
# new_array = []
# for i in P2 :
#     i = np.reshape(i,(1,2))
#     array_1 = distance(P0,P1,i)
#     new_array.append(array_1)
#
# new_array_2 = np.array(new_array)
# print(new_array_2)



# z = np.random.randint(0,2,5)
# print(z)
# np.logical_not(z , out=z)
# print(z)
# np.negative(z, out=z)
# print(z)


# 80 need to look back

# z = np.arange(1,15,dtype=np.uint32)
# print(z)
# r = stride_tricks.as_strided(z,(11,4),(4,4))
# print(r)


# z = np.random.uniform(0,100,(10,10))
# U , S , V  = np.linalg.svd(z)
# rank = np.sum(S > 1e-10)
# print(rank)


# z = np.random.randint(0,10,10)
# print(np.bincount(z).argmax())


# Z = np.random.randint(0,5,(10,10))
# print(Z)
# n = 3
# i = 1 + (Z.shape[0]-n)
# j = 1 + (Z.shape[1]-n)
# C =  np.lib.stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
# print(C)


# class Symetric(np.ndarray):
#     def __setitem__(self, key, value):
#         i,j = key
#         super(Symetric,self).__setitem__((i,j),value)
#         super(Symetric,self).__setitem__((j,i),value)
#
# def symetric(array):
#     return np.asarray(array + array.T - np.diag(array.diagonal())).view(Symetric)
#
# S = symetric(np.random.randint(0,10,(5,5)))
# S[2,3] = 42
# print(S)


# p, n = 10, 20
# M = np.ones((p,n,n))
# V = np.ones((p,n,1))
# S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
# print(S)



# z = np.ones((16,16))
# k = 4
# s = np.add.reduceat(np.add.reduceat(z,np.arange(0,16,k) , axis=0 ), np.arange(0,16,k),axis=1)
# print(s)



# def game_of_Life(Z):
#     N = (Z[0:-2, 0:-2] + Z[0:-2, 1:-1] + Z[0:-2, 2:] +
#          Z[1:-1, 0:-2] + Z[1:-1, 2:] +
#          Z[2:, 0:-2] + Z[2:, 1:-1] + Z[2:, 2:])
#
#     birth = (N == 3) & (Z[1:-1, 1:-1] == 0)
#     survive = ((N == 2) | (N == 3)) & (Z[1:-1, 1:-1] == 1)
#     Z[...] = 0
#     Z[1:-1, 1:-1][birth | survive] = 1
#     return Z
# np.set_printoptions(threshold=np.inf)
# Z = np.random.randint(0,2,(50,50))
# for i in range(50):
#     X = game_of_Life(Z)
#     Z = X
# print(Z)



# z = np.arange(1000)
# np.random.shuffle(z)
# n = 5
# print(z[np.argsort(z)[-n:]])
# print(z[np.argpartition(-z,n)[:n]])



# def cartesian(arrays) :
#     arrays = [np.asarray(a) for a in arrays]
#     shape  = ( len(x) for  x in arrays )
#
#     ix  = np.indices(shape, dtype=int)
#     ix = ix.reshape(len(arrays),-1).T
#
#     for n ,arr in enumerate(arrays) :
#         ix[:,n] = arrays[n][ix[:,n]]
#
#     return ix
#
# print(cartesian([[1,2,3],[4,5,6],[7,8,9]]))



# Z = np.array([("Hello" , 2.5 , 3), ("World" , 3.6 , 2)])
# # print(Z)
# R = np.core.records.fromarrays(Z.T, names='col1, col2, col3' , formats ='S8,f8,i8')
# print(R)

# import numpy as np
# import timeit
#
# Z= np.random.rand(500000)
# Z1 = np.power(Z,3)
# time_1 = timeit.timeit('Z1',globals= globals(), number=10)
# print(time_1)
#
# Z2 = Z*Z*Z
# time_2 = timeit.timeit('Z2' ,globals=globals(), number=10)
# print(time_2)
#
# Z3 = np.einsum('i,i,i->i',Z,Z,Z)
# time_3 = timeit.timeit('Z3', globals=globals(), number=10)
# print(time_3)



# A = np.random.randint(0,5,(5,10))
# B = np.random.randint(0,5,(2,2))
# C = (A[...,np.newaxis,np.newaxis] == B)
# rows = np.where(C.any((3,1)).all(1))[0]
# print(rows)



# Z = np.random.randint(0,5,(10,3))
# print(Z)
# E = np.all(Z[:,1:] == Z[:,:-1], axis=1)
# U = Z[~E]
# print(U , "\n")
#
# V = Z[Z.max(axis=1)!=Z.min(axis=1)]
# print(V)



# I = np.array([0,1,2,3,15,16,32,64,128] , dtype=np.uint8)
# print(np.unpackbits(I[:,np.newaxis],axis = 1))


# A = np.random.uniform(0,10,10)
# print(A,"\n")
# B = np.random.uniform(0,10,10)
# print(B , "\n")
#
# print(np.einsum('i->',A))
# print("\n")#sumation over every element
# print(np.einsum('i,i->i',A,B))  #multiplication by element wise
# print("\n")
# print(np.einsum('i,i',A,B)) #multiplication and summation
# print("\n")
# print(np.einsum('i,j->ij',A,B)) #output a 2-D array





