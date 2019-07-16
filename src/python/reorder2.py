def fib(n):
    a, b = 1, 2
    for _ in range(n):
        yield a
        a, b = b, a + b

def multd(p, d):
    return [d*i for i in p]

def patstr(p):
    return "\"pattern\":" + str(p)

def delstr(delta):
    out = "\"delta\":\""
    for d in delta:
       out = out + str(d) + ","
    return out[0:len(out)-1] + "\""

def namestr(name):
    return "\"name\":\"" + name + "\""

def entry(a, b, c):
    print("  {{\"count\":{}, {}, {}, {}}},".format(2**24, a, c, b))

fiblist = list(fib(4))

print("[")
for d in fiblist:
    p = [0, 1, 2, 3, 4, 5, 6, 7]
    pd = multd(p,d)
    delta = [8*d]
    entry(patstr(pd), delstr(delta), namestr("overlap0"))

for d in fiblist:
    p = [0, 4, 1, 5, 2, 6, 3, 7]
    pd = multd(p,d)
    delta = [8*d]
    entry(patstr(pd), delstr(delta), namestr("overlap0"))

for d in fiblist:
    p = [0, 8, 1, 9, 2, 10, 3, 11]
    pd = multd(p,d)
    delta = [4*d,12*d]
    entry(patstr(pd), delstr(delta), namestr("overlap0"))

for d in fiblist:
    p = [0, 16, 1, 17, 2, 18, 3, 19]
    pd = multd(p,d)
    delta = [2*d,2*d,2*d,26*d]
    entry(patstr(pd), delstr(delta), namestr("overlap0"))

for d in fiblist:
    p = [0, 32, 1, 33, 2, 34, 3, 35]
    pd = multd(p,d)
    delta = [d, d, d, d, d, d, d, 57*d]
    entry(patstr(pd), delstr(delta), namestr("overlap0"))
print("]")

exit(0)
