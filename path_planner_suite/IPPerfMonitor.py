# coding: utf-8

"""
This code is part of the course 'Introduction to robot path planning' (Author: Bjoern Hein). It is based on the slides given during the course, so please **read the information in theses slides first**

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import pandas
import time


class IPPerfMonitor(object):
    "Decorator that keeps track of the number of times a function is called."

    __instances = {}

    def __init__(self, f):

        self.__f = f
        self.data = []
        IPPerfMonitor.__instances[f] = self

    def __call__(self, *args, **kwargs):

        starttime = time.time()
        ret = self.__f(*args, **kwargs)
        endtime = time.time()
        element = {'args': args, 'kwargs': kwargs, "retVal": ret, "time": endtime-starttime}
        self.data.append(element)

        return ret

    def _showargs(self, *fargs, **kw):
        print("T: enter {} with args={}, kw={}".format(self.__f.__name__, str(fargs), str(kw)))

    def __get__(self, instance, owner):
        from functools import partial
        return partial(self.__call__, instance)

    @staticmethod
    def dataFrame():
        "Return a dict of {function: # of calls} for all registered functions."
        result = []
        for f in IPPerfMonitor.__instances:

            for dataElement in IPPerfMonitor.__instances[f].data:
                context = dict({"name": f.__name__})
                context.update(dataElement)
                result.append(context)
        return pandas.DataFrame.from_dict(result)  # type: ignore

    @staticmethod
    def clearData():
        "Clear data"
        for f in IPPerfMonitor.__instances:
            del IPPerfMonitor.__instances[f].data[:]
