def genDat(nom=["manu", "augustin"], n=100):
    data = []
    for r, nm in enumerate(nom):
        for i in range(n):
            data1 = []
            f = open("../data/"+nm+"/mesh"+str(i)+".dat", "r")

            line = f.readline()
            line = line.split("[")
            line2 = []
            for x in line:
                line2.append(x.split("]"))
            line3 = []
            for x in line2:
                for y in x:
                    line3.append(y.split(","))
            for x in line3:
                for y in x:
                    line.append(y.split(" "))
            ban = [' ', '\n', '[', ']', ',', '-']
            for x in line:
                for y in x:
                    if y not in ban and len(y) > 0:
                        data1.append(int(y))

            f.close()
            data.append([data1, r])
    return(data)
