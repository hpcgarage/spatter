import  math

def powers(start, end, pow=2):
    low = math.floor(math.log(start, pow))
    high = math.ceil(math.log(end, pow))

    return [pow**i for i in range(low, high)]

def powers_exclusive(start, end, pow=2):
    low = math.ceil(math.log(start, pow))
    high = math.floor(math.log(end, pow))

    return [pow**i for i in range(low, high)]

def fib(n):
    phi = (1 + 5 ** 0.5) / 2
    return int((phi**n - (-phi)**(-n)) / math.sqrt(5))
    
def fibs(start, stop):
    low = 0
    while fib(low) < start:
        low = low+1
    if fib(low) > start:
        low = low-1

    high = low
    while fib(high) < stop:
        high = high+1

    return [fib(i) for i in range(low, high+1)]

def fibs_exclusive(start, stop):
    low = 0
    while fib(low) < start:
        low = low+1

    high = low
    while fib(high) < stop:
        high = high+1
    if fib(high) > stop:
        high = high-1

    return [fib(i) for i in range(low, high+1)]

def clean(a):
    b = list(set(a))
    b.sort()
    return b

def get_sizes(start, stop):
    a = powers(start, stop)
    b = fibs(start, stop)
    return clean(a + b)


