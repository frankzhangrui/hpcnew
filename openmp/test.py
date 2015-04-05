serial=[]
static=[]
dynamic=[]
guided=[]

for line in open("serial_output","r"):
    serial.append(float(line))
for line in open("static_parallel_output","r"):
    static.append(float(line))
for line in open("dynamic_parallel_output","r"):
    dynamic.append(float(line))
for line in open("guided_parallel_output","r"):
    guided.append(float(line))





assert len(static)==len(serial)
error=[abs(static[i]-serial[i]) for i in range(len(serial))]
print "absolute error per element of static scheduling is: "+ str(sum(error)/len(error))



assert len(guided)==len(serial)
error=[abs(guided[i]-serial[i]) for i in range(len(serial))]
print "absolute error per element of guided scheduling is: "+ str(sum(error)/len(error))

assert len(dynamic)==len(serial)
error=[abs(dynamic[i]-serial[i]) for i in range(len(serial))]
print "absolute error per element of dynamic scheduling is: "+ str(sum(error)/len(error))
	

