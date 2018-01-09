def puzzle_1_1(inputstr):
    extentstr=inputstr+inputstr[0]
    print(sum([int(extentstr[i]) for i in range(len(inputstr)) if extentstr[i]==extentstr[i+1]]))

def puzzle_1_2(inputstr):
    extentstr=inputstr*2
    print(sum([int(extentstr[i]) for i in range(len(inputstr)) if extentstr[i+int(0.5*len(inputstr))]==extentstr[i]]))

def puzzle_2_1(list_of_list_of_num):
    return sum([max(x)-min(x) for x in list_of_list_of_num ])

def puzzle_2_2(list_of_list_of_num):
    print sum([puzzle_2_2_row(row) for row in list_of_list_of_num ])


def puzzle_2_2_row(row):
    row=sorted(row)
    for i1 in range(1,len(row)):
        for i2 in range(i1):
            if row[i1]%row[i2] ==0:
                return row[i1]/row[i2]

def puzzle_3_1(i):
    import math
    sides = math.ceil(math.sqrt(i))
    round =(sides -1)/2
    total_round = (sides **2) - (sides-2)**2
    nr = i - (sides-2)**2
    print
    anticlock= round + ((nr-min([-1, -round ]))%(max([total_round/4,1])))
    clock = round + ((-(nr-min([-1, -round ])))%(max([total_round/4,1])))
    return int(min([clock,anticlock]))

def puzzle_3_2(rounds,endval):
    import numpy
    grid_rad =rounds+1
    dir=[[[-1,0],1],[[0,-1],2],[[1,0],2],[[0,1],3]]
    grid=numpy.zeros([(grid_rad)*2+1,(grid_rad)*2+1])
    zero = [grid_rad,grid_rad]
    grid[zero[0],zero[1]]=1
    x1 =zero[0]
    x2 =zero[1]+1
    for iter in xrange(rounds):
        for direction in dir:
            for step in range(direction[1]+2*(iter)):
                #print direction[0], x1,x2,numpy.sum(grid[x1-1:x1+1,x2-1:x2+1])
                solution=numpy.sum(grid[x1-1:x1+2,x2-1:x2+2])
                if solution>endval:
                    return solution
                grid[x1,x2] = solution
                x1+=direction[0][0]
                x2+=direction[0][1]


def puzzle_day_4_1():
    with  open("C:\Users\Michael\Downloads\input_day_4.txt") as filevar :
        data = [x.split() for x in filevar.readlines()]

    return len([1 for x in data if len(set(x)) == len(x)])

def puzzle_day_4_2():
    with  open("C:\Users\Michael\Downloads\input_day_4.txt") as filevar :
        data = [x.split() for x in filevar.readlines()]
    data=[["".join(sorted(x)) for x in y] for y in data]

    return len([1 for x in data if len(set(x)) == len(x)])

def puzzle_day_5_1():
        with  open("C:\Users\Michael\Downloads\input_day_5.txt") as filevar :
            jumplist = [int(x) for x in filevar.read().split()]
        ix = 0
        jumps = 0
        while (ix >-1) and (ix < len(jumplist)):
            offset = jumplist[ix]
            if offset>2:
                jumplist[ix]=jumplist[ix]-1
            else:
                jumplist[ix]=jumplist[ix]+1
            ix+=offset
            jumps+=1
        print jumps

def puzzle_day_6_2():
        with  open("C:\Users\Michael\Downloads\input_day_6.txt") as filevar :
            bankslist = [int(x) for x in filevar.read().split()]
        #bankslist = [0,2,3,4]
        solutiondict={" ".join([str(x) for x in bankslist]): 0}
        rounds =0
        while True:
            highest=max(bankslist)
            highestidx = bankslist.index(highest)
            bankslist[highestidx]=0
            for dist in range(1,highest+1):
                bankslist[(highestidx+dist)%len(bankslist)]+=1
            print bankslist
            rounds+=1
            strsol=" ".join(str(x) for x in bankslist)
            if strsol in solutiondict:
                print rounds
                print rounds-solutiondict[strsol]
                break
            else:
                solutiondict[strsol]=rounds


def puzzle_day_7_1():
        with  open("C:\Users\Michael\Downloads\input_day_7.txt") as filevar :
            graphtext =filevar.readlines()

        parent_daughter_dict={}
        parent_weight_dict={}
        daughter_parent_dict={}
        for line in graphtext:
            daughters=""
            if len(line.split(" -> "))>1:
                [parent,daughters]=line.split(" -> ")
            else:
                parent=line
            [name,weight]=parent.split()
            weight = int(weight[1:len(weight)-1])
            daughters = [x.strip() for x in daughters.split(", ")]
            parent_weight_dict[name]=weight
            parent_daughter_dict[name]=daughters
            for daughter in daughters:
                daughter_parent_dict[daughter]=name
        print set(parent_daughter_dict)-set(daughter_parent_dict)

def puzzle_day_7_2():

        def isbalanced(weightdict,daughterdict,name):
            if name in daughterdict:
                return len(set([weightdict[daughter] for daughter in daughterdict[name]]))<2
            else:
                return True

        with  open("C:\Users\Michael\Downloads\input_day_7.txt") as filevar :
            graphtext =filevar.readlines()
        import copy
        parent_daughter_dict={}
        parent_weight_dict={}
        daughter_parent_dict={}
        for line in graphtext:

            if len(line.split(" -> "))>1:
                [parent,daughters]=line.split(" -> ")
                daughters = [x.strip() for x in daughters.split(", ")]
            else:
                daughters=[]
                parent=line
            [name,weight]=parent.split()
            weight = int(weight[1:len(weight)-1])
            parent_weight_dict[name]=weight
            parent_daughter_dict[name]=daughters
            for daughter in daughters:
                daughter_parent_dict[daughter]=name
        total_weight=copy.deepcopy(parent_weight_dict)

        for name in parent_weight_dict:
            weight = parent_weight_dict[name]
            daughter=name
            while True:
                if daughter in daughter_parent_dict:
                    daughter=(daughter_parent_dict[daughter])
                    total_weight[daughter]+=weight
                else:
                    break

        for name in parent_daughter_dict:
            if not (isbalanced(total_weight,parent_daughter_dict,name)):
                if all([isbalanced(total_weight,parent_daughter_dict,daughter) for daughter in parent_daughter_dict[name]]):
                    totalweights=[total_weight[daughter] for daughter in parent_daughter_dict[name]]
                    weights=[parent_weight_dict[daughter] for daughter in parent_daughter_dict[name]]
                    print totalweights
                    print weights

def puzzle_day_8_1():
    import operator
    import collections
    opdict={
        "inc" : operator.add,
        "dec" : operator.sub,
        "==" : operator.eq,
        "<": operator.lt,
        ">": operator.gt,
        ">=": operator.ge,
        "<=": operator.le,
        "!=": operator.ne}

    registerdict=collections.defaultdict(int)

    with  open("C:\Users\Michael\Downloads\input_day_8.txt") as filevar :
        lines =filevar.readlines()
    #lines=[
    #"b inc 5 if a > 1",
    #"a inc 1 if b < 5",
    #"c dec -10 if a >= 1",
    #"c inc -20 if c == 10"]

    all_time=[]
    for line in lines:
        #rint line
        var1, op, int1, dummy, condvar, cond_op, int2 = line.split()
        if opdict[cond_op](registerdict[condvar],int(int2)):
            registerdict[var1]=opdict[op](registerdict[var1],int(int1))
            all_time.append(registerdict[var1])
    print max(registerdict.values())
    print max(all_time)


def puzzle_day_9_1():
    with open("C:\Users\Michael\Downloads\input_day_9.txt") as filevar :
        charlist =list(filevar.read())

    filteredlist=[]
    ix=0
    while ix <len(charlist):
        if charlist[ix] == "!":
            ix+=2
        else:
            filteredlist.append(charlist[ix])
            ix+=1

    filteredlist2=[]
    ix=0
    garbage=False
    garbcount=0
    while ix <len(filteredlist):
        if garbage:
            if filteredlist[ix] ==">":
                garbage=False
            else:
                garbcount+=1
        else:
            if filteredlist[ix] =="<":
                garbage=True
            else:
                filteredlist2.append(filteredlist[ix])
        ix+=1
    print garbcount

    totalscore=0
    score_each=1
    for letter in filteredlist2:
        if letter == "{":
            totalscore+=score_each
            score_each+=1
        if letter == "}":
            score_each-=1

    print totalscore



def puzzle_day_10_1():
    with open("C:\Users\Michael\Downloads\input_day_10.txt") as filevar :
        instruct_list =[int(x) for x in filevar.read().split(",")]

    #instruct_list = [3, 4, 1, 5]
    listlen = 256
    pos=0
    skip=0
    numberlist = range(listlen)
    for swaplength in instruct_list:
        subsection=numberlist[pos:min(pos+swaplength,listlen)] +numberlist[0:max((pos+swaplength)-listlen,0)]
        subsection=subsection[::-1]
        for itemnr in range(len(subsection)):
            numberlist[(pos+itemnr)%listlen]=subsection[itemnr]
        pos+=swaplength
        pos+=skip
        pos=pos%listlen
        skip+=1
        print numberlist[0]*numberlist[1]




def puzzle_day_10_2():
    def clean_hex(a):
        b=str(hex(a))
        if len(b)==3:
            return b[0]+b[2]
        else:
            return b[2:4]

    from functools import reduce
    with open("C:\Users\Michael\Downloads\input_day_10.txt") as filevar :
        text =filevar.read().strip()
    print text
    instruct_list = [ord(x) for x in text ]
    #instruct_list=[ord(x) for x in "AoC 2017"]
    instruct_list += [17, 31, 73, 47, 23]
    listlen = 256
    pos=0
    skip=0
    numberlist = range(listlen)
    for iter in range(64):
        for swaplength in instruct_list:
            subsection=numberlist[pos:min(pos+swaplength,listlen)] +numberlist[0:max((pos+swaplength)-listlen,0)]
            subsection=subsection[::-1]
            for itemnr in range(len(subsection)):
                numberlist[(pos+itemnr)%listlen]=subsection[itemnr]
            pos+=swaplength
            pos+=skip
            pos=pos%listlen
            skip+=1
            skip = skip % listlen

    densehash=[]
    for hashnr in range(16):
        densehash.append(reduce(lambda x, y: x^y,numberlist[hashnr*16 : (hashnr+1)*16]))
    print "".join([clean_hex(x)[2:4] for x in [64,7,255]])
    hashstr = "".join([clean_hex(x) for x in densehash])
    print hashstr


def puzzle_day_11():

    with open("C:\Users\Michael\Downloads\input_day_11.txt") as filevar :
        dir_list =filevar.read().strip().split(",")
    from collections import Counter

    dir_to_offset={
        "ne":(1,-1),
        "n":(0,-1),
        "s":(0,1),
        "nw":(-1,0),
        "se":(1,0),
        "sw":(-1,1),
    }
    def dist_from_coords(coords):
        x=coords[0]
        y=coords[1]
        dist=0
        if abs(x+y) < max([abs(x),abs(y)]):
            dist = max([abs(x),abs(y)])
        else:
            dist = abs(x)+abs(y)
        return dist

    count_dict=Counter()
    coords = [0,0]
    maxdist=0
    for dir in dir_list:
        count_dict[dir]+=1
        coords[0] += dir_to_offset[dir][0]
        coords[1] += dir_to_offset[dir][1]
        dist = dist_from_coords(coords)
        if dist > maxdist:
            maxdist = dist

    print maxdist

def puzzle_day_12_1():
    with open("C:\Users\Michael\Downloads\input_day_12.txt") as filevar :
        lines =filevar.readlines()

    def splitline(line):
        name, connections = line.split(" <-> ")
        connections = [int(x) for x in connections.split(", ")]
        name=int(name)
        return (name,connections)
    connectdict={}

    for line in lines:
        name,connections = splitline(line)
        connectdict[name]=connections

    done=set([])
    todo=set([0])

    while len(todo)>0:
        next=todo.pop()
        if next not in done:
            done.add(next)
            for item in connectdict[next]:
                todo.add(item)

    print len(done)

def puzzle_day_12_2():
    with open("C:\Users\Michael\Downloads\input_day_12.txt") as filevar :
        lines =filevar.readlines()

    def splitline(line):
        name, connections = line.split(" <-> ")
        connections = [int(x) for x in connections.split(", ")]
        name=int(name)
        return (name,connections)
    connectdict={}

    for line in lines:
        name,connections = splitline(line)
        connectdict[name]=connections

    nodes_not_reached=set(connectdict.keys())
    groups = 0
    while len(nodes_not_reached)>0:
        done=set([])
        todo=set([list(nodes_not_reached)[0]])

        while len(todo)>0:
            next=todo.pop()
            if next not in done:
                done.add(next)
                for item in connectdict[next]:
                    todo.add(item)
        nodes_not_reached=nodes_not_reached-done
        groups+=1

    print groups

def puzzle_day_13_1():
    with open("C:\Users\Michael\Downloads\input_day_13.txt") as filevar :
        lines =filevar.readlines()

    layerdict={}
    for line in lines:
        [layer, depth] = [int(x) for x in line.split(": ")]
        layerdict[layer]=depth

    delay=0
    while True:
        delay+=1
        score=0
        solution=True
        for layer in layerdict:
            if (layer+delay)%(layerdict[layer]+max([layerdict[layer]-2,0])) ==0:
                solution=False
                break
        if solution:
            break
    print delay

def knothash(instr):
    def clean_hex(a):
        b=str(hex(a))
        if len(b)==3:
            return b[0]+b[2]
        else:
            return b[2:4]

    from functools import reduce
    instruct_list = [ord(x) for x in instr ]
    #instruct_list=[ord(x) for x in "AoC 2017"]
    instruct_list += [17, 31, 73, 47, 23]
    listlen = 256
    pos=0
    skip=0
    numberlist = range(listlen)
    for iter in range(64):
        for swaplength in instruct_list:
            subsection=numberlist[pos:min(pos+swaplength,listlen)] +numberlist[0:max((pos+swaplength)-listlen,0)]
            subsection=subsection[::-1]
            for itemnr in range(len(subsection)):
                numberlist[(pos+itemnr)%listlen]=subsection[itemnr]
            pos+=swaplength
            pos+=skip
            pos=pos%listlen
            skip+=1
            skip = skip % listlen

    densehash=[]
    for hashnr in range(16):
        densehash.append(reduce(lambda x, y: x^y,numberlist[hashnr*16 : (hashnr+1)*16]))
    hashstr = "".join([clean_hex(x) for x in densehash])
    return hashstr


def puzzle_day_14_1():
    def num2bin(hexletter):
        num=int(hexletter,16)
        raw_bin=str(bin(num))
        raw_bin=raw_bin[1:len(raw_bin)]
        bin_num="0"*(4-len(raw_bin))+raw_bin
        return bin_num

    input="flqrgnkx"
    count=0
    for row in range(128):
        rowinput=input+"-"+str(row)
        output = knothash(rowinput)
        binout="".join([num2bin(hexletter) for hexletter in output])
        count+=binout.count("1")
    print count

def puzzle_day_14_2():

    def num2bin(hexletter):
        num=int(hexletter,16)
        raw_bin=str(bin(num))
        raw_bin=raw_bin[2:len(raw_bin)]
        bin_num="0"*(4-len(raw_bin))+raw_bin
        return bin_num

    input="oundnydw"
    spacelist=[]
    for row in range(128):
        rowinput=input+"-"+str(row)
        output = knothash(rowinput)
        binout="".join([num2bin(hexletter) for hexletter in output])
        spacelist.append(binout)

    usedsquareset=set([])
    for rowix in range(128):
        for colix in range(128):
            if spacelist[rowix][colix]=="1":
                usedsquareset.add((rowix,colix))

    def get_ajacent(coords):
        dirs=[[1,0],[-1,0],[0,1],[0,-1]]
        coords = [(coords[0]+dir[0],coords[1]+dir[1]) for dir in dirs]
        return coords

    regions=0
    import copy
    valid = copy.deepcopy(usedsquareset)
    while len(usedsquareset)>0:
        done=set([])
        nextone = usedsquareset.pop()
        usedsquareset.add(nextone)
        todo=set([nextone])

        while len(todo)>0:
            this_one=todo.pop()
            if this_one not in done:
                done.add(this_one)
                for ajacent in get_ajacent(this_one):
                    if ajacent in valid:
                        todo.add(ajacent)
        regions+=1
        usedsquareset -=done

    print regions

    for row in range(8):
        print spacelist[row][0:8]


def puzzle_day_15_1():
    def properbin(innr):
        strbin=str(bin(innr))[2:]
        strbin=("0"*(16-min([len(strbin),16])))+strbin
        return strbin

    class numgenerator:
        def __init__(self,initial,factor):
            self.number=initial
            self.factor=factor
        def take_next(self):
            self.number=(self.number*self.factor)%2147483647
            binary=properbin(self.number)
            return binary[len(binary)-16:len(binary)]

    gen1=numgenerator(883,16807)
    gen2=numgenerator(879,48271)
    score=0
    for iter in xrange(4*10**7):
        if gen1.take_next() == gen2.take_next():
            score+=1
    print score

def puzzle_day_15_2():
    def properbin(innr):
        strbin=str(bin(innr))[2:]
        strbin=("0"*(16-min([len(strbin),16])))+strbin
        return strbin

    class numgenerator:
        def __init__(self,initial,factor,filt):
            self.number=initial
            self.factor=factor
            self.filt=filt
        def take_next(self):
            self.number=(self.number*self.factor)%2147483647
            if self.number%self.filt ==0:
                binary=properbin(self.number)
                return binary[len(binary)-16:len(binary)]
            else:
                return self.take_next()

    gen1=numgenerator(883,16807,4)
    gen2=numgenerator(879,48271,8)
    score=0
    for iter in xrange(5*10**6):
        if gen1.take_next() == gen2.take_next():
            score+=1
    print score

def puzzle_16_1():
    order="a	b	c	d	e	f	g	h	i	j	k	l	m	n	o	p".split()
    n=16
    with open("C:\Users\Michael\Downloads\input_day_16.txt") as filevar :
        commands =filevar.read().split(",")
    for command in commands:
        if command[0]=="s":
            shift=int(command[1:])
            order=order[(n-shift):n] + order[0:n-shift]
        if command[0]=="x":
            [idx1,idx2]=[int(x) for x in command[1:].split("/")]
            val1=order[idx1]
            val2=order[idx2]
            order[idx1]=val2
            order[idx2]=val1
        if command[0]=="p":
            idx1=order.index(command[1])
            idx2=order.index(command[3])
            val1=order[idx1]
            val2=order[idx2]
            order[idx1]=val2
            order[idx2]=val1
        print order

    print "".join(order)

def puzzle_16_2():
    order="a	b	c	d	e	f	g	h	i	j	k	l	m	n	o	p".split()
    original=order[:]
    n=16
    with open("C:\Users\Michael\Downloads\input_day_16.txt") as filevar :
        commands =filevar.read().split(",")

    passes=0
    while (order != original) or passes==0:
        for command in commands:
            if command[0]=="s":
                shift=int(command[1:])
                order=order[(n-shift):n] + order[0:n-shift]
            if command[0]=="x":
                [idx1,idx2]=[int(x) for x in command[1:].split("/")]
                val1=order[idx1]
                val2=order[idx2]
                order[idx1]=val2
                order[idx2]=val1
            if command[0]=="p":
                idx1=order.index(command[1])
                idx2=order.index(command[3])
                val1=order[idx1]
                val2=order[idx2]
                order[idx1]=val2
                order[idx2]=val1
        passes+=1
    print "cyclelen", passes
    for i in xrange(1000000000%passes):
        for command in commands:
            if command[0]=="s":
                shift=int(command[1:])
                order=order[(n-shift):n] + order[0:n-shift]
            if command[0]=="x":
                [idx1,idx2]=[int(x) for x in command[1:].split("/")]
                val1=order[idx1]
                val2=order[idx2]
                order[idx1]=val2
                order[idx2]=val1
            if command[0]=="p":
                idx1=order.index(command[1])
                idx2=order.index(command[3])
                val1=order[idx1]
                val2=order[idx2]
                order[idx1]=val2
                order[idx2]=val1



    print "".join(order)

def puzzle_17_1():

    linkedlist={0:0}
    current=0
    stepsize=394
    for year in xrange(1,2018):
        for move in range(stepsize):
            current=linkedlist[current]
        nextone=linkedlist[current]
        linkedlist[current]=year
        linkedlist[year]=nextone
        current=year

    print linkedlist[current]

def puzzle_17_2():
    linkedlist={0:0}
    current=0
    stepsize=394
    for year in xrange(1,50000001):
        for move in range(stepsize):
            current=linkedlist[current]
        nextone=linkedlist[current]
        linkedlist[current]=year
        linkedlist[year]=nextone
        current=year

    while current !=0:
        current=linkedlist[current]
    print linkedlist[current]



def puzzle17_2_2():
    current=0
    stepsize=394
    last_after_0=0
    for year in xrange(1,50000001):
        current=(current+stepsize)%year
        if current == 0:
            last_after_0 = year
        current+=1 # if it was 0, after insterion it is 1
    print last_after_0


def puzzle18_1():
    from collections import defaultdict
    with open("C:\Users\Michael\Downloads\input_day_18.txt") as filevar :
        commands =filevar.readlines()

    registers=defaultdict(int)
    last_played="none"
    commandix=0
    while (commandix>=0) and (commandix<len(commands)):

        command=commands[commandix].split()
        print command
        if command[0]=="snd":
            last_played=registers[command[1]]
        elif command[0] == "set":
            try:
                registers[command[1]]= int(command[2])
            except ValueError:
                registers[command[1]]= registers[command[2]]

        elif command[0]=="add":
            try:
                registers[command[1]]= registers[command[1]] + int(command[2])
            except ValueError:
                registers[command[1]]= registers[command[1]] + registers[command[2]]

        elif command[0] == "mul":
            try:
                registers[command[1]]= registers[command[1]] * int(command[2])
            except ValueError:
                registers[command[1]]= registers[command[1]] * registers[command[2]]

        elif command[0] == "mod":
            try:
                registers[command[1]]= registers[command[1]] % int(command[2])
            except ValueError:
                registers[command[1]]= registers[command[1]] % registers[command[2]]

        elif command[0] == "rcv":
            if registers[command[1]] != 0:
                registers[command[1]]=last_played
                if last_played!=0:
                    print last_played

                    break
        elif command[0] == "jgz":
            itemlist=[]
            for item in command[1:3]:
                try:
                    itemlist.append(int(item))
                except:
                    itemlist.append(registers[item])

            if itemlist[0]>0:
                commandix+= itemlist[1]-1
        else:
            print command
        commandix+=1
        print registers


def puzzle18_2():
    from collections import defaultdict, deque
    with open("C:\Users\Michael\Downloads\input_day_18.txt") as filevar :
        commands =filevar.readlines()

    class tunegen():
        def __init__ (self,startnr):
            self.ix=0
            self.registers=defaultdict(int)
            self.registers["p"]=startnr
            self.received=deque()
            self.count_send=0
            self.waiting=False

        def setpartner(self,partner):
            self.partner=partner

        def receive(self,value):
            self.received.append(value)
            self.waiting=False

        def run(self):
            registers=self.registers
            #print self.received
            while True:
                command=commands[self.ix].split()
                #print registers
                #print command
                if command[0]=="snd":
                    self.partner.receive(registers[command[1]])
                    self.count_send+=1
                elif command[0] == "set":
                    try:
                        registers[command[1]]= int(command[2])
                    except ValueError:
                        registers[command[1]]= registers[command[2]]

                elif command[0]=="add":
                    try:
                        registers[command[1]]= registers[command[1]] + int(command[2])
                    except ValueError:
                        registers[command[1]]= registers[command[1]] + registers[command[2]]

                elif command[0] == "mul":
                    try:
                        registers[command[1]]= registers[command[1]] * int(command[2])
                    except ValueError:
                        registers[command[1]]= registers[command[1]] * registers[command[2]]

                elif command[0] == "mod":
                    try:
                        registers[command[1]]= registers[command[1]] % int(command[2])
                    except ValueError:
                        registers[command[1]]= registers[command[1]] % registers[command[2]]

                elif command[0] == "rcv":
                    if len(self.received)>0:
                        registers[command[1]]=self.received.popleft()
                    else:
                        self.waiting=True
                        break

                elif command[0] == "jgz":
                    itemlist=[]
                    for item in command[1:3]:
                        try:
                            itemlist.append(int(item))
                        except:
                            itemlist.append(registers[item])

                    if itemlist[0]>0:
                        self.ix+= itemlist[1]-1
                else:
                    print command
                self.ix+=1
                #print registers

    prog1=tunegen(0)
    prog2=tunegen(1)
    print prog1.registers
    print prog2.registers
    prog1.setpartner(prog2)
    prog2.setpartner(prog1)

    while True:
        #print "next round"
        prog1.run()
        prog2.run()
        #print len(prog1.received),len(prog2.received)
        if prog1.waiting and prog2.waiting:
            break
    print prog1.count_send
    print prog2.count_send

def puzzle_day_19_1():
    def newcoord(coords,delta):
        return (coords[0]+delta[0],coords[1]+delta[1])

    def validatecoords(coords,mazemap):
        if all([coords[0]>-1,
                    coords[1]>-1,
                    coords[0]<len(mazemap),
                    coords[1]<len(mazemap[0])]):
            return mazemap[coords[0]][coords[1]] !=" "


    with open("C:\Users\Michael\Downloads\input_day_19.txt") as filevar :
        mazemap =filevar.readlines()

    enterance_col=mazemap[0].index("|")
    dirs = [(1,0),(0,1),(0,-1),(-1,0)]
    dirdict={
        (1,0):[(1,0),(0,1),(0,-1)],
        (0,1):[(0,1),(1,0),(-1,0)],
        (0,-1):[(0,-1),(1,0),(-1,0)],
        (-1,0):[(-1,0),(0,1),(0,-1)]}
    currentdir=(1,0) #row,col
    currentcoords=[0,enterance_col]
    stuck=False
    visited = []
    steps=0
    capital_letters=set(map(chr, range(65, 91)))
    while not stuck:
        for optiondir in dirdict[currentdir]:
            nextcoords=newcoord(currentcoords,optiondir)
            if validatecoords(nextcoords,mazemap):
                currentcoords=nextcoords
                currentdir=optiondir
                steps+=1
                currentcharacter=mazemap[currentcoords[0]][currentcoords[1]]
                if currentcharacter in capital_letters:
                    visited.append(currentcharacter)
                break
        else:
            stuck=True
    print "".join(visited)
    print steps+1



def puzzle_day_20_1():

    with open("C:\Users\Michael\Downloads\input_day_20.txt") as filevar :
        particlelist =filevar.readlines()

    min_total_a=float("inf")
    lowest_a_particle=""

    particledict={}
    for particlenr in range(len(particlelist)):
        p,v,a=particlelist[particlenr].split(", ")
        particleproperties={"p":p,"a":a,"v":v}
        for key in particleproperties:
            particleproperties[key]=particleproperties[key].split("=")[1]
            particleproperties[key]=particleproperties[key].strip().strip("<>")
            particleproperties[key] = [int(x) for x in particleproperties[key].split(",")]
        particledict[particlenr]=particleproperties
        total_a=sum([abs(x) for x in particleproperties["a"]])
        if total_a<min_total_a:
            lowest_a_particle=particlenr
            min_total_a=total_a
    print lowest_a_particle
    import collections
    for step in range(1000):
        locationcount=collections.defaultdict(int)
        for particle in particledict:
            particle_obj=particledict[particle]
            particle_obj["v"]=[particle_obj["v"][i]+particle_obj["a"][i] for i in range(3)]
            particle_obj["p"]=[particle_obj["p"][i]+particle_obj["v"][i] for i in range(3)]
            locationcount[(tuple(particle_obj["p"]))]+=1

        for particle in particledict.keys():
            particle_obj=particledict[particle]
            if locationcount[tuple(particle_obj["p"])]>1:
                del particledict[particle]

        print len(particledict)



def puzzle_day_21_1():

    with open("C:\Users\Michael\Downloads\input_day_21.txt") as filevar :
        rulelist =filevar.readlines()

    def flip_h(inputlist):
        horzflip=[x[::-1] for x in inputlist]
        return [x[::-1] for x in inputlist]

    def flip_v(inputlist):
        return inputlist[::-1]

    def rotate_once(inputlist):
        return ["".join([inputlist[i][ii] for i in range(len(inputlist))[::-1] ]) for ii in range(len(inputlist))]


    ruledict={}
    for rule in rulelist:
        rule=rule.split("=>")
        inp  = rule[0].strip().split("/")
        outp = rule[1].strip().split("/")
        for i in range(4):
            ruledict[tuple(inp)]=tuple(outp)
            ruledict[tuple(flip_h(inp))]=tuple((outp))
            ruledict[tuple(flip_v(inp))]=tuple((outp))
            inp=rotate_once(inp)
            outp=outp

    grid = [".#.","..#","###"]
    """grid2=rotate_once(grid)
    grid3=rotate_once(grid2)
    grid4=rotate_once(grid3)
    for gridn in [grid,grid2,grid3,grid4]:
        for row in gridn:
            print row
        print "\n"""

    for i in range(18):
        length=len(grid)
        if length%2==0:
            breaksize=2
        else:
            breaksize=3

        newgrid=[]
        for row in range(length/breaksize):
            for subrow in range(breaksize+1):
                newgrid.append([])
            for col in range(length/breaksize):
                gridrow=range(row*breaksize, (row+1)*breaksize)
                gridcol=range(col*breaksize, (col+1)*breaksize)
                sector= tuple(["".join([grid[rowi][coli] for coli in gridcol]) for rowi in gridrow])
                newsector=ruledict[sector]
                for sectorrow in range(len(newsector)):
                    newgrid[row*(breaksize+1)+sectorrow]+=newsector[sectorrow]
        grid=newgrid

        print "width",len(grid)
        total=0
        for row in newgrid:
            #print row
            for char in row:
                if char=="#":
                    total+=1
        print total
        print "\n"


def puzzle_22_1():
    with open("C:\Users\Michael\Downloads\input_day_22.txt") as filevar :
        maze =filevar.readlines()
    maze=[line.strip() for line in maze]

    #maze = ["..#","#..","..."]

    infectedset=set([])
    weakset=set([])
    flagedset=set([])
    for rownr in range(len(maze)):
        for colnr in range(len(maze[rownr])):
            if maze[rownr][colnr]=='#':
                infectedset.add((rownr,colnr))

    dirs= [(-1,0),#up
           (0,1), #right
           (1,0), #down
           (0,-1)] #left


    currentplace=((len(maze)-1)/2, (len(maze[0])-1)/2)
    print len(maze),len(maze[0]),currentplace
    print infectedset
    currendirnr=0
    infected=0
    for i in xrange(10000000):
        if currentplace in infectedset:
            currendirnr=(currendirnr+1)%4
            infectedset.remove(currentplace)
            flagedset.add(currentplace)

        elif currentplace in weakset:
            weakset.remove(currentplace)
            infectedset.add(currentplace)
            infected+=1

        elif currentplace in flagedset:
            flagedset.remove(currentplace)
            currendirnr=(currendirnr+2)%4

        else:
            currendirnr=(currendirnr-1)%4
            weakset.add(currentplace)


        #print dirs[currendirnr]
        currentplace = (currentplace[0]+dirs[currendirnr][0],
                        currentplace[1]+dirs[currendirnr][1])

    print infected



def puzzle_23_1():
    from collections import defaultdict
    with open("C:\Users\Michael\Downloads\input_day_23.txt") as filevar :
        commands =filevar.readlines()

    registers=defaultdict(int)
    registers["a"]=1
    commandix=0
    total=0
    while (commandix>=0) and (commandix<len(commands)):
        command=commands[commandix].split()
        if command[0] == "set":
            try:
                registers[command[1]]= int(command[2])
            except ValueError:
                registers[command[1]]= registers[command[2]]

        elif command[0]=="sub":
            try:
                registers[command[1]]= registers[command[1]] - int(command[2])
            except ValueError:
                registers[command[1]]= registers[command[1]] - registers[command[2]]

        elif command[0] == "mul":
            try:
                registers[command[1]]= registers[command[1]] * int(command[2])
            except ValueError:
                registers[command[1]]= registers[command[1]] * registers[command[2]]

            total+=1

        elif command[0] == "jnz":
            itemlist=[]
            for item in command[1:3]:
                try:
                    itemlist.append(int(item))
                except:
                    itemlist.append(registers[item])

            if itemlist[0]!=0:
                commandix+= itemlist[1]-1
        else:
            print command
        commandix+=1
    print registers["h"]




def puzzle_23_2():
    bi=67*100+100000
    c=bi+17000
    primes=set([2,3,5])
    for i in range(7,bi,2):
        for ii in primes:
            if i%ii==0:
                break
        else:
            primes.add(i)
            print i
    nonprimes=0
    for b in range(bi, c + 1, 17):
        for ii in primes:
            if b%ii==0:
                nonprimes+=1
                break
        else:
            primes.add(b)
            print "prime",b
    print nonprimes

def puzzle_24_1():
    import collections
    import copy
    with open("C:\Users\Michael\Downloads\input_day_24.txt") as filevar :
        ports =filevar.readlines()
    ports = [x.split("/") for x in ports]
    ports = [[int(xi),int(xii)] for xi,xii in ports]

    def solvenext(optionlist,currentport):
        bestscore=0
        bestlength=0
        for optionnr in range(len(optionlist)):
            if currentport in optionlist[optionnr]:
                new_optionlist=copy.deepcopy(optionlist)
                del new_optionlist[optionnr]
                port = copy.deepcopy(optionlist[optionnr])
                portscore=sum(port)
                del port[port.index(currentport)]
                nextport=port[0]
                nextscore, nextlength = solvenext(new_optionlist,nextport)
                thisscore=nextscore+portscore
                thislength=nextlength+1
                if thislength>bestlength:
                    bestlength=thislength
                    bestscore=thisscore
                elif thislength==bestlength:
                    if thisscore>bestscore:
                        bestscore=thisscore


        return [bestscore,bestlength]
    print solvenext(ports,0)

def puzzle_25_1():
    with open("C:\Users\Michael\Downloads\input_day_25.txt") as filevar :
        rules =filevar.readlines()

    endpoint=12629077
    rulesdict={}
    rules=rules[3:]
    i=0
    while i<len(rules):
        s=rules[i].split()[-1][0]
        i+=2
        w0=rules[i].split()[-1].strip(".")
        w0=int(w0)
        i+=1
        m0=rules[i].split()[-1].strip(".")
        if m0=="right":
            m0=1
        else:
            m0=-1
        i+=1
        s0=rules[i].split()[-1].strip(".")
        i+=2
        w1=rules[i].split()[-1].strip(".")
        w1=int(w1)
        i+=1
        m1=rules[i].split()[-1].strip(".")
        if m1=="right":
            m1=1
        else:
            m1=-1
        i+=1
        s1=rules[i].split()[-1].strip(".")
        rulesdict[s]={0:[w0,m0,s0],1:[w1,m1,s1]}
        i+=2

    ones=set([])
    pos=0
    state="A"
    for iter in xrange(endpoint):#endpoint):
        if pos in ones:
            [w,m,s] = rulesdict[state][1]
            if w == 0:
                ones.remove(pos)
            pos+=m
            state=s
        else:
            [w,m,s] = rulesdict[state][0]
            if w == 1:
                ones.add(pos)
            pos+=m
            state=s
        #print ones
    print len(ones)

