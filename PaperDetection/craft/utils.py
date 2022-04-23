import sys

def getSubTotal(a):
        if len(a)<2:
            return 0
        j=1
        sum=0
        found_len=0
        found_sum=a[j]
        while j<len(a):
            sum=sum+a[j-1]
            i=j
            while(i<len(a)):
                if sum==a[i]:
                    if found_len<j:
                        found_len=j
                        found_sum=sum
                i+=1
            j+=1
        print("found it", a[:found_len], found_sum)
        return found_sum

a = [45000,55000,100000,5000,25000,10000,500000,240000,260000]
b = [45000,55000,100000,5000,25000,10000,20000,260000]
c = [45000,55000,100000,5000,25000,10000,20000,260000,45000,55000,100000,5000,25000,10000,20000,260000,1500000,1040000,460000]
d = [10000,10000]
total = getSubTotal(a)
total = getSubTotal(b)
total = getSubTotal(c)
total = getSubTotal(d)
