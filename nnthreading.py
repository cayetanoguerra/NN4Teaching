import threading
import numpy as np
import time

import multiprocessing

print "Multi: ", multiprocessing.cpu_count()

def worker():
    """thread worker function"""
    print 'Worker'
    t1 = time.time()
    r = 0.
    aux = 0.
    for i in xrange(5000000):
        aux = np.sqrt(i)
        r += aux
    print r
    t2 = time.time()
    print "Tiempo worker: ", t2-t1
    return

def worker2():
    """thread worker function"""
    r = 0.
    aux = 0.
    for i in xrange(5000000):
        aux = np.sqrt(i)
        r += aux
    print r
    return


threads = []
for i in range(3):
    t = threading.Thread(target=worker)
    threads.append(t)
    t.start()


print "Secuencial"
t1 = time.time()
r1=worker2()
r2=worker2()
r3=worker2()
print r1 == r2
print r1 == r3
t2 = time.time()
print "Tiempo sin threading: ", t2-t1