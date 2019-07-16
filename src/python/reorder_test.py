def fib(n):
    a, b = 5, 8
    for _ in range(n):
        yield a
        a, b = b, a + b

fiblist = list(fib(6))
pats=[]
for d in fiblist:
    pats.append("\"pattern\":[0, 1, 2, 3, {}, {}, {}, {}], \"delta\":{}, \"name\":\"Shift0\"".format(d,d+1,d+2,d+3, 2*d))
for d in fiblist:
    pats.append("\"pattern\":[0, 1, 2, {}, 3, {}, {}, {}], \"delta\":{}, \"name\":\"Shift1\"".format(d,d+1,d+2,d+3, 2*d))
for d in fiblist:
    pats.append("\"pattern\":[0, 1, {}, 2, {}, 3, {}, {}], \"delta\":{}, \"name\":\"Shift2\"".format(d,d+1,d+2,d+3, 2*d))
for d in fiblist:
    pats.append("\"pattern\":[0, {}, 1, {}, 2, {}, 3, {}], \"delta\":{}, \"name\":\"Shift3\"".format(d,d+1,d+2,d+3, 2*d))


base="{{\"count\":{}, {}}},"

out = "[\n"
for p in pats:
    out = out + " " + base.format(2**24, p) + "\n"

out = out + "]"
print(out)
