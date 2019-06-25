# Python wrapper for the Spatter Benchmark

# Goals: Parse Json File
# Generate configs
# Initialize Spatter, pass configs, 
import copy 
import json
input="""[{'pattern':'UNIFORM:8:8', 'kernel':'Gather', 'length':[123,444], 'delta':[(1,2),4]}, 
          {'pattern':(10,20,30),'length':[2**i for i in range(10)]}]"""

input2="""[{'pattern':"UNIFORM:8:8"}]"""

def main(): 

    dat = eval(input2)
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

