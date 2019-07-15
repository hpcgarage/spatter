def fib(n):
    a, b = 1, 2
    for _ in range(n):
        yield a
        a, b = b, a + b

fiblist = list(fib(10))
pats=[]
for d in fiblist:
    pats.append("\"pattern\":[0, 1, 2, 3, {}, {}, {}, {}], \"delta\":{}, \"name\":Easy".format(d,d+1,d+2,d+3, 2*d))
for d in fiblist:
    pats.append("[\"pattern\":0, {}, 1, {}, 2, {}, 3, {}],\"delta\":{}, \"name\":Hard".format(d,d+1,d+2,d+3, 2*d))
