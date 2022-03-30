from gettext import npgettext
from numpy import float64, random, array


def unison_shuffled_copies(dat, cl):
    assert len(dat) == len(cl)
    p = random.permutation(len(dat))
    nd = []
    nc = []
    for x in p:
        nd.append(dat[x])
        nc.append(cl[x])

    return nd, nc


def genDat(nom=["manu", "pierre"], n=160):
    data = []
    clss = []
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
                        data1.append(float64(y))

            f.close()
            if len(data1) == 3744:  # a changer, je veux tous le meme longuer

                data.append(data1)
                clss.append(r)

    d, c = unison_shuffled_copies(data, clss)

    return([d, c])


def genUsable(file):

    data1 = []
    f = open("../data/"+file, "r")

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
                data1.append(float64(y))

    f.close()
    if len(data1) == 3744:  # a changer, je veux tous le meme longuer

        return (array(data1).reshape(1, 48, 78, 1))
