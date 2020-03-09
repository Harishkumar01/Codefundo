import pandas as pd
import math

def dis(p,q):
    l=0
    for i in range(0,len(p)):
        l+=(p[i]-q[i])**2
    return math.sqrt(l)

def kMeans(k,a,b,c,l):
    #find the distance between l and all the points
    #put the cluster number into cluster
    cluster=[]
    for i in range(0,len(a)):
        m=100000
        t=[]
        t.append(float(a[i]))
        t.append(float(b[i]))
        t.append(float(c[i]))
        index=-1
        for j in range(0,len(l)):
            x=dis(t,l[j])
            if(x<m):
                m=x
                index=j
        cluster.append(index)
        
    #find mean for each cluster
    m=[]
    A=[]
    B=[]
    C=[]
    for i in range(0,k):
        m.append(0)
        A.append(0)
        B.append(0)
        C.append(0)
    for i in range(0,len(a)):
        for j in range(0,k):
            if(cluster[i]==j):
                A[j]+=a[j]
                B[j]+=b[j]
                C[j]+=c[j]
                m[j]+=1
    
    L=[]
    for i in range(0,k):
        t=[]
        t.append(A[j]/m[j])
        t.append(B[j]/m[j])
        t.append(C[j]/m[j])
        L.append(t)
    
    flag=0
    for i in range(0,k):
        for j in range(0,len(l[i])):
            if(int(l[i][j])!=int(L[i][j])):
              flag=1
    
    if(flag==1):
        kMeans(k,a,b,c,L)
        
    else:
        print("The final clusters are : \n")
        for i in range(0,k):
            print(l[i])
            print("\n")

data=pd.read_csv('iris.csv',usecols=[0,1,2])

a=data.iloc[:,0].values
b=data.iloc[:,1].values
c=data.iloc[:,2].values

k=int(input("k = "))

#taking first k points randomly
l=[]
t=[]
for i in range(0,k):
    t.append(float(a[i]))
    t.append(float(b[i]))
    t.append(float(c[i]))
    l.append(t)
    t=[]
#now l contains the cluster point
    
kMeans(k,a,b,c,l)
