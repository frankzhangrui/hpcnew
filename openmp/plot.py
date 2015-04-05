import matplotlib.pyplot as plt
size=[100,1000,10000,100000]
serial=[0.000514,0.025703,2.571077,256.453655]
static=[0.027302, 0.026964, 0.436134, 31.137050]
guided=[0.019706,0.026730,0.394147,29.860100]
dynamic=[0.025486,0.022454,0.235969, 21.219162]



speedup_static=[serial[i]/static[i] for i in range(len(static))]
speedup_guided=[serial[i]/guided[i] for i in range(len(static))]
speedup_dynamic=[serial[i]/dynamic[i] for i in range(len(static))]

plt.xlabel('sizes')
plt.ylabel('time')
plt.loglog(size,serial,label='serial')
plt.loglog(size,static,label='static')
plt.loglog(size,guided,label='guided')
plt.loglog(size,dynamic,label='dynamic')
plt.legend(loc='upper left')
plt.show()




plt.xlabel('sizes')
plt.ylabel('speed_up')
plt.loglog(size,speedup_static,label='static')
plt.loglog(size,speedup_guided,label='guided')
plt.loglog(size,speedup_dynamic,label='dynamic')
plt.legend(loc='upper left')
plt.show()