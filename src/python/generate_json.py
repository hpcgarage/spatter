# Python wrapper for the Spatter Benchmark

# Goals: Parse Json File
# Generate configs
# Initialize Spatter, pass configs, 
import copy 
import json
input="""[{'pattern':'UNIFORM:8:8', 'kernel':'Gather', 'length':[123,444], 'delta':[(1,2),4]}, 
          {'pattern':(10,20,30),'length':[2**i for i in range(10)]}]"""

input2="""[{'pattern':"UNIFORM:8:8"}]"""

input3="""[{'pattern':["UNIFORM:8:{}:NR".format(i) for i in range(8)]+["UNIFORM:8:{}:{}".format(i, i*4) for i in range(10)], 'count':2**24} ]"""
            
input4="""[{'pattern':["UNIFORM:8:{}:{}".format(i, i*8) for i in range(10)], 'count':2**24, 'name':'Reuse 0%'}, 
           {'pattern':["UNIFORM:8:{}:{}".format(i, i*6) for i in range(10)], 'count':2**24, 'name':'Reuse 25%'},
           {'pattern':["UNIFORM:8:{}:{}".format(i, i*4) for i in range(10)], 'count':2**24, 'name':'Reuse 50%'},
           {'pattern':["UNIFORM:8:{}:{}".format(i, i*2) for i in range(10)], 'count':2**24, 'name':'Reuse 75%'},
           {'pattern':["UNIFORM:8:{}:{}".format(i, i*0) for i in range(10)], 'count':2**24, 'name':'Reuse 100%'}
          ]"""

input5="""[{'pattern':(0, 1, 2, 3, d, d+1, d+2, d+3), 'count':2**24, 'name':'Good'},
           {'pattern':(0, d, 1, d+1, 2, d+2, 3, d+3), 'count':2**24, 'name':'Bad'}
          ]"""

def main(): 

    dat = eval(input4)
    change = True

    while change:
        change = False
        for d in list(dat):
            for key, value in d.items():
                if (type(value) == list):
                    dat.remove(d)
                    l = len(value)
                    tmp = []
                    for i in range(l):
                        tmp.append(copy.deepcopy(d))
                        tmp[i].pop(key)
                        tmp[i][key] = value[i]
                    change = True
                    dat = dat + tmp
                    break
    print(json.dumps(dat))

if __name__ == "__main__":
    main()

