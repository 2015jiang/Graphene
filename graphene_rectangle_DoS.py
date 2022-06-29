# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 17:45:47 2022

@author: ZJ
"""

import numpy as np
import matplotlib.pyplot as plt
from math import *
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号


Nx=20
Ny=20
t_NN_inplane=2.8#同一层最近邻AB之间的hopping
on_site_energy=0

matrix1=np.array([[0.0, 0.0],[t_NN_inplane, 0.0]])
matrix2=np.array([[0.0, t_NN_inplane],[0.0, 0.0]])
h00=np.kron(np.eye(2), np.array([[0.0, t_NN_inplane],[t_NN_inplane, 0.0]]))+np.kron(np.eye(2, k=1), matrix1)+np.kron(np.eye(2, k=-1), matrix1.T)
h01=np.array([[0.0, t_NN_inplane, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, t_NN_inplane, 0.0]])
# h000=np.array([[0.0, 0.0,0.0, 0.0],[0.0, 0.0,0.0, 0.0],[0.0, 0.0,0.0, 0.0],[0.0, 0.0,0.0, 0.0]])
h000=np.array([[0.0, t_NN_inplane],[t_NN_inplane, 0.0]])
h001=np.array([[t_NN_inplane,0.0],[0.0, 0.0],[0.0, 0.0],[0.0, t_NN_inplane]])
h001T=np.array([[t_NN_inplane,0.0, 0.0, 0.0],[0.0, 0.0, 0.0, t_NN_inplane]])

H00=np.zeros((4*Nx+2,4*Nx+2))
H01=np.zeros((4*Nx+2,4*Nx+2))

H00[0:4*Nx,0:4*Nx]=np.kron(np.eye(Nx), h00)+np.kron(np.eye(Nx, k=1), h01)+np.kron(np.eye(Nx, k=-1), h01.T)
H00[4*Nx:4*Nx+2,4*Nx:4*Nx+2]=h000
H00[4*Nx-4:4*Nx,4*Nx:4*Nx+2]=h001
H00[4*Nx:4*Nx+2,4*Nx-4:4*Nx]=h001T

# 周期边界条件
# H00[0:4,4*Nx-4:4*Nx]=h01
# H00[4*Nx-4:4*Nx,0:4]=h01.T

def Hamiltonian_Graphenemodel_square(Nx,Ny,t_NN_inplane,on_site_energy):

    for i in np.arange(0, 4*Nx, 4):
        H01[i,i+3] = t_NN_inplane
        H=np.kron(np.eye(Ny), H00)+np.kron(np.eye(Ny, k=1), H01)+np.kron(np.eye(Ny, k=-1), H01.T)
        # 周期边界条件
        # H[0:(4*Nx+2),(4*Nx+2)*(Ny-1):(4*Nx+2)*Ny]=H01
        # H[(4*Nx+2)*(Ny-1):(4*Nx+2)*Ny,0:(4*Nx+2)]=H01.T
        
    
    return H



                    

# print(H)


   
    

def main():
    plot_precision = 0.01  # 画图的精度
    Fermi_energy_array = np.arange(-9.0, 9.0, plot_precision)  # 计算中取的费米能Fermi_energy组成的数组
    dim_energy = Fermi_energy_array.shape[0]   # 需要计算的费米能的个数
    total_DOS_array = np.zeros((dim_energy))   # 计算结果（总态密度total_DOS）放入该数组中
    h = Hamiltonian_Graphenemodel_square(Nx,Ny,t_NN_inplane,on_site_energy)  # 体系的哈密顿量
    dim = h.shape[0]   # 哈密顿量的维度
    i0 = 0
    for Fermi_energy in Fermi_energy_array:
        print(Fermi_energy)  # 查看计算的进展情况
        green = np.linalg.inv((Fermi_energy+0.1j)*np.eye(dim)-h)   # 体系的格林函数
        total_DOS = -np.trace(np.imag(green))/pi    # 通过格林函数求得总态密度
        total_DOS_array[i0] = total_DOS   # 记录每个Fermi_energy对应的总态密度
        i0 += 1
    sum_up = np.sum(total_DOS_array)*plot_precision    # 用于图像归一化
    plt.figure(figsize=(6,3.7),dpi=300)
    plt.plot(Fermi_energy_array, total_DOS_array/sum_up, linewidth=2.0, linestyle= '-')   # 画DOS(E)图像
    plt.xlabel('费米能',fontsize=24)
    plt.ylabel('总态密度',fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()


if __name__ == '__main__':
    main()