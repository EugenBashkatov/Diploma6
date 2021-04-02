# This is a sample Python script.

# Token:
# cd5c0f2b8bb97f0cd4e46dd6f0e5647922a163d3
import statistics
import sys
import os
import math
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto, norm, poisson, binom, expon
import stem_plot as st
import datetime as dt
import seaborn as sns
import utilities as ut
from matplotlib.backends.backend_pdf import PdfPages

import mymodule as my
import distributions as dst
# numm=-math.inf
# if numm<10000000:print (math.atan(numm))
# exit()
import xlsxwriter
ONLY_NORMAL = True
NEED_NORMALIZE =False
DEBUG=True
DEBUG1 = False
DEBUG2 = False
DEBUG3 = False
DEBUG4 = False
DEBUG5 = False
MAX_STAT= 15
start_date = "1981-01-01"
end_date = "1981-01-15"


def get_logger(name=__file__, file='log.txt', encoding='utf-8'):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s] %(filename)s:%(lineno)d %(levelname)-8s %(message)s')
    # os.remove("log.txt")
    fh = logging.FileHandler(file, encoding=encoding, mode='w')
    #    fh = logging.FileHandler(file, encoding=encoding)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(formatter)
    log.addHandler(sh)

    return log


log = get_logger()

if(DEBUG): log.debug("start")


def print_hi(name):
    # Use a breakpoint in the code get_k below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def load_datalist_csv(filename='daily-min-temperatures.csv'):
    input_file_name = filename
    full_df = pd.read_csv(
        input_file_name,
        names=[
            'Date',
            'MinTemp'],
        index_col=[0])
    return full_df


def load_datalist_filter_csv(start_date,end_date,filename='daily-min-temperatures.csv'):
    df=load_datalist_csv().loc[start_date:end_date].copy(deep=True)
    return  df['MinTemp'].tolist()
if __name__ == '__main__':
    print_hi('PyCharm')
    data_list=load_datalist_filter_csv(start_date,end_date)
    aaa=len(data_list)
    # TODO # почиctить фильтрацию
    # input_file_name = 'daily-min-temperatures.csv'
    # input_file_name = 'daily-min-temperatures.csv'
    # full_df = pd.read_csv(
    #     input_file_name,
    #     names=[
    #         'Date',
    #         'MinTemp'],
    #     index_col=[0])
    # start_date = "1981-01-01"
    # end_date = "1981-01-31"
    # df = full_df.loc[start_date:end_date].copy(deep=True)
    # data_list = full_df['MinTemp'].tolist()[:MAX_STAT]
# TODO делаем несмещенную вборку
# TODO НОРМИРОВАНИЕ ВЫБОРКИ
if NEED_NORMALIZE:
    my_mean=statistics.mean(data_list)
    data_list= list(map(float,data_list))[:MAX_STAT]
    #data_list=list(full_df['MinTemp'])[:MAX_STAT]
    #ddd=list(full_df['MinTemp'])[:MAX_STAT]
    for i in range(0,len(data_list)-1):
        data_list[i]-=my_mean
    my_mean1=statistics.mean(data_list)
# TODO сделать одномерный массив data_list заполнить для замены дат
max_dim = len(data_list)

# max_dim = len(data_list)
out_data_list_name = "data_list from {0} to {1} length {2}.csv".format(
    start_date, end_date, max_dim)
new_ind = np.arange(0, max_dim)
fig, ax=plt.subplots()
ax.stem(new_ind, data_list,label='temp 31')
ax.set_title('title')
# plt.show()
# TODO
# st.stemplot(new_ind, data_list)

pd.DataFrame(data_list).to_csv(out_data_list_name)
#full_df.to_csv("full_df.csv")
mymean=statistics.mean(data_list)
data_norm=[]
for i in range(0,len(data_list)):
    data_norm.append(data_list[i]-mymean)

# st.stem2(new_ind,data_list,'native',data_norm,'normal',SAVEPP=True)


#TODO Испытательный полигон
if DEBUG:
    print(ut.get_index_of_max(data_list))
#    print(ut.get_data_by_index(df,ut.get_index_of_max(data_list)))
    print("Mean = {}, Var = {}".format(ut.statistics(data_list)[0],ut.statistics(data_list)[1]))


# ------------------------------------DEFS--------------------------------------------


def get_k(x0, x1, data_list):
    y0 = data_list[x0]
    y1 = data_list[x1]

    k = (y1 - y0) / (x1 - x0)

    return k

def print_graph_array(graph_array):
    for ind in range(0, len(graph_array)):
        print(ind, ":", graph_array[ind])


def is_cluster_found(x0, x1, x2, data_list, max_dim, control_k):
    vis_k = get_k(x0, x1, data_list)
    try:
        vis_k_next = get_k(x0, x2, data_list)
    except IndexError as e:
        vis_k_next = vis_k
    ind = np.argmax([control_k, vis_k, vis_k_next])
    return vis_k, vis_k_next, ind


def get_last_x(d_cluster_chain):
    values_list = list(d_cluster_chain.values())
    last_cluster = values_list[len(d_cluster_chain) - 1]
    first_x = last_cluster[0]
    last_x = last_cluster[len(last_cluster) - 1]

    return first_x, len(last_cluster)

def build_A(data_list):

    A = []
    for i in range(0,len(data_list)-2):
        A.append(data_list[i+1]/data_list[i])
    meanA = statistics.mean(A)

    controlA = []
    diffA = []
    for i in range(0,len(data_list)-2):
        controlA.append(meanA*data_list[i])

    return meanA,A-controlA

def build_normal_datalist(par_size):
    #normal_data_list = []
    normal_data_list = list(norm.rvs(size = par_size))
    _my_mean=statistics.mean(normal_data_list)
    return normal_data_list

def autoregress_list(start_point,fi,par_size):

    norm_list = build_normal_datalist(par_size)
    ret_autoregress_list = []
    ret_autoregress_list.append(start_point)
    # xi+1 = fi*xi+norm_eps

    for i in range(0,len(norm_list)-1):
        ret_autoregress_list.append(fi*ret_autoregress_list[i]+norm_list[i])

    return ret_autoregress_list
# print(build_A(data_list))
# print("normal_data_list = {}".format(build_normal_datalist()))
# print(len(autoregress_list(5,0.6)))


# d_cluster_mean[x1] = statistics.mean(list(map(lambda item: data_list[item], interior_list)))
# ------------------------------------DEFS--------------------------------------------

logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')

start_point = 0
""""""
start_time=dt.datetime.now()
d_cluster_chain = {0: [0]}
d_cluster_length = {0: 1}
d_cluster_mean = {}
d_cluster_var = {}

def build_graph_with_clusters(start_point, data_list, DEBUG=False):

    x0 = start_point  # начало кластера
    x1 = x0 + 1  # рабочая вершина
    x2 = x1 + 1  # контрольная вершина
    cluster_size = 1
    # if(DEBUG): log.debug(("x0={} x1={}, x2={} control01={} ".format(x0, x1, x2, control_k))

    array_k = []
    interior_list = []

    # -----------------------------------------------
    # Start main algoriithm
    # -----------------------------------------------
    d_cluster_begin = {}
    control_k = get_k(x0, x1, data_list)

    x0 = start_point
    x1 = x0 + 1

    p_cluster_begin = []
    # d_cluster_length={0:2}
    # d_cluster_chain = {0: [0]}

    #     log.debug("firsst={}".format(x1 - 1))
    while True:

        x2 = x1 + 1
        cluster_len = 1

        while x2 <= max_dim - 1:  # TODO

            vis_k, vis_k_next, ind = is_cluster_found(
                x0, x1, x2, data_list, max_dim, control_k)

            # if(DEBUG): log.debug(("x0={} x1={}, x2={} ind={} control_k={} control_2={}".format(x0, x1, x2, ind, control_k, control_2))
            # cluster end x0 start new cluster x1=x2

            if ind == 2:  # найдено окончание кластера:

                if(DEBUG): log.debug(" найдено окончание кластера в x0={} x1={} x2={} sublist={} clu_len={}".format(x0, x1, x2,
                                                                                                         interior_list,                                                                                  cluster_len))
                if x1 == x0 + 1:  # # TODO

                    if(DEBUG): log.debug(
                        '2= найден первый кластер {} {} {}'.format(p_cluster_begin, d_cluster_chain, d_cluster_begin))
                    if(DEBUG): log.debug(
                        "* найдено окончание кластера в x0={} x1={} x2={} sublist={} clu_len={}".format(x0, x1, x2,
                                                                                                        interior_list,
                                                                                                        cluster_len))
                    cluster_len = x2 - x1 + 1
                    p_cluster_begin.append(x0)
                    if(DEBUG): log.debug("p_cluster_begin={}".format(p_cluster_begin))
                    # d_cluster_chain[x0] = interior_list.insert(x0, x1)
                    d_cluster_chain[x0] = [x0, x0 + 1]
                    if(DEBUG): log.debug("d_cluster_chain={}".format(d_cluster_chain))
                    # d_cluster_length[x0] = cluster_len
                    # if(DEBUG): log.debug("***d_cluster_length={}".format(d_cluster_length))

                else:
                    cluster_len = x2 - x1
                    p_cluster_begin.append(x1)
                    # d_cluster_length[x1] = cluster_len
                    d_cluster_chain[x1] = interior_list
                    # TODO Затычка подсчета интериор листа

                # st.rays_plot(x0, data_list[x0], x2, data_list[x2])

                # TODO Fix patch #1
                if x2 >= 2:

                    interior_list.insert(0, x1)

                # x0 = x1  # новое начало кластера x0=x1

                x1 = x2
                x2 += 1
                # TODO PATCH #3
                cluster_len = 1

                control_k = get_k(x0, x1, data_list)
                if x2 >= max_dim:
                    control2 = control_k

                # if(DEBUG): log.debug(( "x0={} x1={}, x2={} ind={} control_k={} control_2={}".format( x0, x1, x2, ind, control_k, control_2))
                # if(DEBUG): log.debug("5 Cluster start={} interior_list={} cluster_len={}".format(x1, interior_list, cluster_len))

                interior_list = []

                x2 = x1 + 1

                if DEBUG1:
                    print("DEBUG_1:From= ", x0, "To= ", x1, "k= ", vis_k, "Next= ", x2, "next_k=", vis_k_next,
                          "cluster_size=", cluster_size, "ind=", ind, "cluster_len=", cluster_len)
            if ind != 2:  # continue cluster: x1 собирание цепочки вершин для класиера, x1 входит в текущий кластер

                interior_list.append(x2)
                cluster_len += 1
                # if(DEBUG): log.debug("5 Cluster start={} interior_list={}".format(x1, interior_list))

                x2 += 1

                # if DEBUG1: print("DEBUG_1:From= ", x0, "To= ", x1, "k= ", vis_k, "Next= ", x2, "next_k=", vis_k_next,
                #                  "cluster_size=", cluster_size, "ind=", ind, "cluster_len=", cluster_len)
            array_k.append([x0, x1, x2, vis_k, vis_k_next, cluster_len])

        if x2 > (max_dim - 1):
            cluster_len = x2 - x1
            p_cluster_begin.append(x1)
            # d_cluster_length[x1] = cluster_len
            interior_list.insert(0, x1)
            d_cluster_chain[x1] = interior_list
            print("------------------------------------------------------------")
            break

    # for i in range(0, len(p_cluster_begin)):
    #     cur_cluster = p_cluster_begin[i]
    #     cur_chain = d_cluster_chain[cur_cluster]
    #     sum = 0
    #
    #     for j in range(0, len(cur_chain)):
    #
    #         par1 = cur_chain[j]
    #         sum = sum + data_list[par1]
    #
    #
    #     if DEBUG:
    #         log.debug("i=={} begin={} length={} chain={}".format(i, p_cluster_begin[i], d_cluster_length[cur_cluster],
    #                                                      d_cluster_chain[cur_cluster]))
    #         log.debug("d_cluster_chain = ", d_cluster_chain, "len_d_cluster_chain = ", len(d_cluster_chain))

        # print("cur_cluster={} length={} chain={}".format(cur_cluster, d_cluster_length[cur_cluster],d_cluster_chain[cur_cluster]))
    first_x, last_len = get_last_x(d_cluster_chain)
    figs = plt.get_fignums()
    print(figs)


    for i in d_cluster_chain.keys():
        d_cluster_length[i] = len(d_cluster_chain[i])
    print("d_cluster_chain = ", d_cluster_chain)
    print("d_cluster_length_test = ", d_cluster_length)
    return first_x, x2, last_len


def draw_pic():
    fig,ax=plt.subplots()
    # st.stemplot(new_ind, data_list, SAVEPP=True, pdfname='TEST.jpg')
    ax.stem(new_ind,reg_data_list)
    for i in d_cluster_chain:
        print(d_cluster_chain[i])
        xstart = d_cluster_chain[i][0]
        ystart = reg_data_list[xstart]

        # for j in range(1,len(d_cluster_chain[i]) -1 ):
        for j in range(1,len(d_cluster_chain[i])):
            xend = d_cluster_chain[i][j]
            yend = reg_data_list[xend]
            # ax.rays_plot(xstart,ystart,xend,yend)
            ax.plot([xstart, xend],[ystart,yend],'-g')
            print("xstart = {}, ystart = {}, xend = {}, yend = {}".format(xstart,ystart,xend,yend))
    for i in range(0,len(d_cluster_chain)-1):
        x_start = list(d_cluster_chain.keys())[i]
        x_next = list(d_cluster_chain.keys())[i+1]
        y_start = reg_data_list[x_start]
        y_next = reg_data_list[x_next]
        ax.plot([x_start, x_next], [y_start, y_next], '-r')


    plt.show()
    return True


# while start_point < max_dim - 2:
#     start_point, last_x, last_len = build_graph_with_clusters(start_point, data_list, max_dim, False)
#     if(DEBUG): log.debug("000**first={} lsat={} max_dim={} last_len={}".format(start_point, last_x, max_dim, last_len))


# TODO Просто раскомментировать для парето
# data_list = list(pareto.rvs(0.2, size = 128))
# data_list = list(norm.rvs(size = 128))
# data_list = list(poisson.rvs(0.1, size = 128))
# data_list = np.arange(binom.ppf(0.01, 128, 0.6),binom.ppf(0.99, 128, 0.6))
# data_list = list(expon.rvs(size = 128))

#-------------------------------------------------------------
dens = []
# TODO Fix d_cluster_chain. Dont change
if ONLY_NORMAL:
    #data_list = autoregress_list(data_list[0],0.9,MAX_STAT)
    max_dim = len(data_list)
    dfi=0.1
    fi=0
    while fi<0.2:
        start_point = 0
        fi+=dfi
        reg_data_list = autoregress_list(0.7, fi, MAX_STAT)
        # data_list = list(norm.rvs(size = 256))
        #if DEBUG:log.debug("fi = {}".format(fi))
        max_dim = len(reg_data_list)
        new_ind = np.arange(0, max_dim)
        # st.stemplot(new_ind, data_list,SAVEPP=True,pdfname='TEST.jpg')

        d_cluster_chain = {0: [0]}
        print("*****************fi = {} ***********************".format(fi))
        while start_point < max_dim - 2:

            next_point, last_x, last_len = build_graph_with_clusters(start_point, reg_data_list, False)
            # print("d_cluster_chain = ", d_cluster_chain)
            # print("d_cluster_length = ", d_cluster_length)
            a = next_point
            cond= (a == start_point)
            if cond:
                print("Start_point = {}, cond = {}".format(start_point, cond))
                start_point = 0
                break
            else:

                start_point = a
                print("||||||||||||||||||||||||STARTPOINT = {} ||||||||||||||||||||||".format(start_point))


            if(DEBUG1): log.debug("000***first={} lsat={} max_dim={} last_len={}".format(start_point, last_x, max_dim, last_len))
        # st.stemplot(new_ind, data_list, suptitle="Normal distribution", title='fi = {}'.format(fi),
        #             pdfname='fi{}.pdf'.format(fi), SAVEPP=True)

        # plt.show()
        if DEBUG:
            Density=len(d_cluster_chain) / len(reg_data_list)
            dens.append(Density)
            log.debug("fi = {}, Density={} num clustters={} len datalist={}".format(fi,Density,len(d_cluster_chain),len(data_list)))
    draw_pic()
    # print("d_cluster_chain = ",d_cluster_chain)
    # print("d_cluster_length = ",d_cluster_length)
if DEBUG:print("len data_list={}".format(len(data_list)))
# d_cluster_mean[x1] = statistics.mean(list(map(lambda item: data_list[item], interior_list)))

#TODO Испытательные полигон №2

print(np.mean(dens))
elapsed_time=dt.datetime.now()-start_time
print("elapsed_time={}".format(dt.datetime.now()-start_time))
print("END")
max_len = sum(d_cluster_length.values())
# print("cluster_length={} total_len={} : max_dim={}".format(d_cluster_length,max_len,max_dim))
if DEBUG:
    print("start_point = {}, total_len={} : max_dim={}".format(start_point, max_len, max_dim))

print("Density = {}".format(len(d_cluster_chain) / len(data_list)))


plt.show()
aaa=plt.get_fignums()
figs = plt.get_figlabels()
exit()

# ---------------------------------  Version 2.0 ------------------------------
