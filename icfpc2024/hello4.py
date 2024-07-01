BASE=94
def convert(s):
    s=list(s)
    s.reverse()
    num=0
    for i, c in enumerate(s):
        n = ord(c)-ord("!")
        num += n*(BASE**i)
    return num

s1="$>"
s2="1~s:U@"
s3="#"

num1=convert(s1)
num2=convert(s2)
num3=convert(s3)
print(num1,num2,num3)
num=num1*num2+num3
print(num)
# num=15818151

d='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!"#$%&\'()*+,-./:;<=>?@[\]^_`|~ \n'
ans=""
for i in range(10,-1,-1):
    q=num//(BASE**i)
    if q>0:ans+=d[q]
    num -= q*BASE**i
print(ans)