from multiprocessing import Pool
import time

changed = 0

def f(x):
    global changed
    changed = 1
    print(x, changed)

if __name__ == '__main__':
    pool = Pool(processes=1)              # Start a worker processes.
    result = pool.apply_async(f, [1])
    pool.close()
    pool.join()

    time.sleep(1)
    print(changed)