#_*_coding:utf-8_*_
'''
@project: work
@author: exudingtao
@time: 2020/6/24 5:07 下午
'''

import threading
import time, datetime
import multiprocessing
#简而言之,一个程序至少有一个进程,一个进程至少有一个线程.
#Python由于有全锁局的存在（同一时间只能有一个线程执行），并不能利用多核优势
#CPU密集型的用多进程，
#IO密集性的可以利用io阻塞等待的空闲时间来用多线程
#定义线程函数
def print_time(thread_name):
    for i in range(3):
        now = datetime.datetime.now()
        print(now, thread_name)
        time.sleep(1)

#不带线程处理的程序 15秒
# for i in range(5):
#     threadname = "threadName" + str(i)
#     print_time(threadname)
#
# #带线程处理的函数
# #运行发现，不带线程处理的程序和线程处理的程序运行顺序是一样的 15秒
# for i in range(5):
#     threadname = "threadName" + str(i)
#     t = threading.Thread(target=print_time(threadname))
#     t.start()

if __name__ == "__main__":

    # 进程处理
    # for i in range(5):
    #     threadname = "threadName" + str(i)
    #     p = multiprocessing.Process(target=print_time, args=(threadname,))
    #     p.start()
    # 进程池处理
    '''
    由于设置了进程并发的数量为4，所以，前三秒执行的都是前四个进程的内容（每个进程执行完需要三秒），
    进程5只能在前四个进程执行完成之后，才开始执行。总耗时6秒。
    设置为5，则3秒
    '''
    pool = multiprocessing.Pool(processes=4)
    for i in range(5):
        threadname = "threadName" + str(i)
        pool.apply_async(print_time,(threadname,))
    pool.close()
    pool.join()
    print("Sub-process(es) done.")

