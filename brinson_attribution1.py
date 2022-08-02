# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 08:42:47 2019

@author: HP
"""
#last version
#从数据库导入的话，需要去考虑导入fund_code和基准沪深300 以及基金对应行业
import os
import numpy as np
import pandas as pd
from itertools import dropwhile
import warnings
import matplotlib.pyplot as plt
import math 
import time

plt.rcParams["font.sans-serif"]=["SimHei"]#标题为中文
plt.rcParams["axes.unicode_minus"]=False#坐标中有负数
warnings.filterwarnings('ignore') 
wdir = os.path.dirname(__file__)

#数据库
import cx_Oracle
from sqlalchemy import create_engine
conn_string='oracle+cx_oracle://hurz:Gfhurz123@10.88.102.82:1521/?service_name=FINCHINA'
engine = create_engine(conn_string, echo=False)
def get_trade_date_list(flag):
    filepath=os.path.join(wdir,"temp")
    if flag==False:
        data=pd.read_excel(os.path.join(filepath,"data.xlsx"))
        data = data.set_index('zrr')
        return data['ym'].tolist()

    #方法一：create_engine连接数据库
    sql_date = ''' 
        select 
         max(a.trade_days) ym,
         last_day(to_date(max(a.trade_days),'yyyymmdd'))zrr
          from gfwind.AShareCalendar a
        where a.s_info_exchmarket = 'SSE'
        and a.trade_days >= '20090123'
        and a.trade_days <= to_char(sysdate,'yyyymmdd')
        group by trunc(to_date(TRADE_DAYS,'yyyymmdd'),'mm')
        order by trunc(to_date(TRADE_DAYS,'yyyymmdd'),'mm')
    '''
    data = pd.read_sql(sql_date, engine)
    data.to_excel(os.path.join(filepath,"data.xlsx"),encoding="gbk")
    data = data.set_index('zrr')
    
    return data['ym'].tolist()
def load_ind_data(flag=True,ind_type='zx'):
    #date_list = [20220531,20220630]
    filepath=os.path.join(wdir,"temp")
    if flag==False:
        ind_panel=pd.read_excel(os.path.join(filepath,"ind_panel.xlsx"))
        return ind_panel
    date_list = get_trade_date_list(flag)
    ind_panel = pd.DataFrame()
    for sk_date in date_list:
        if ind_type=='zx':
            sql_data = '''
            select to_char(to_date('%s','yyyymmdd'),'yyyy/mm/dd')sk_date,a.s_info_windcode, b.Industriesname
          from gfwind.AShareIndustriesClassCITICS a, gfwind.AShareIndustriesCode b
         where substr(a.citics_ind_code, 1, 4) = substr(b.IndustriesCode, 1, 4)
              --如上条件配合levelnum使用，1级行业截取4位长度，2级行业截取6位，3级行业截取8位
           and b.levelnum = '2' --2表示1级，3表示2级，4表示3级
          and '%s' between a.ENTRY_DT and nvl(remove_dt,'20991231')
            '''%(sk_date,sk_date)
        if ind_type=='sw':
            sql_data = '''
                 select 
            to_char(to_date('%s', 'yyyymmdd'), 'yyyy/mm/dd') sk_date,
            d.s_info_windcode,
                   c.industriesname
              from gfwind.asharedescription d
             inner join gfwind.AShareSWIndustriesClass b
                on (d.s_info_windcode = b.s_info_windcode)
             inner join gfwind.ashareindustriescode c
                on (rpad(substr(b.sw_ind_code, 1, 4), 16, '0') = c.industriescode)
             where '%s' between b.entry_dt and nvl(b.remove_dt, '20991231')
            '''%(sk_date,sk_date)
        data = pd.read_sql(sql_data, engine)
        ind_data=data.pivot(index='s_info_windcode',columns='sk_date',values='industriesname')
        ind_panel = pd.concat([ind_panel, ind_data], axis=1)
    if flag==True:
        ind_panel.to_excel(os.path.join(filepath,"ind_panel.xlsx"),encoding="gbk")
    return ind_panel


#得到行业数据
def get_ind_data(weight,flag=True,ind_type='zx'):
    #industry_zx.csv 股票对应行业
    #industry_dat = pd.read_csv(os.path.join(wdir, 'quote_data', f'industry_{ind_type}.csv'),
                               #encoding='gbk', engine='python', index_col=[0])
    industry_dat=load_ind_data(flag,ind_type='sw')
    industry_dat.columns = pd.to_datetime(industry_dat.columns) #日期格式
    industry_dat = industry_dat.loc[weight.index, weight.columns]#还是自己
    industry_dat = industry_dat.where(pd.notnull(weight), np.nan)#得到对应权重的行业
    return industry_dat

def get_trade_date(date,to_type=0):
    #方法一：create_engine连接数据库
    sql_date = ''' 
        select 
         to_date(max(a.trade_days),'yyyymmdd') ym,
         last_day(to_date(max(a.trade_days),'yyyymmdd'))zrr
          from gfwind.AShareCalendar a
        where a.s_info_exchmarket = 'SSE'
        and a.trade_days >= '20000101'
        and a.trade_days <= to_char(sysdate,'yyyymmdd')
        group by trunc(to_date(TRADE_DAYS,'yyyymmdd'),'mm')
    '''
    data = pd.read_sql(sql_date, engine)
    data = data.set_index('zrr')
    
    #方法2：cx_Oracle 连接数据库
    db=cx_Oracle.connect('hurz','Gfhurz123','10.88.102.82:1521/FINCHINA') 
    cr=db.cursor()
    cr.execute(sql_date)  
    rs=cr.fetchall()
    data1=pd.DataFrame(rs,columns=['ym','zrr'])
    if to_type==0:
        data1 = data1.set_index('zrr')
    else:
        data1 = data1.set_index('ym')
    
    return data1.loc[date][0]

def get_panel_data(date, panel_dir):
    dat = pd.read_csv(os.path.join(panel_dir, f'{str(date)[:10]}.csv'), encoding='gbk',
                      engine='python', index_col=[0])
    return dat
    
def get_stocks_ret(weight, freq='M'):   
    wt = weight.copy()
    '''
    if freq.endswith('M'):
        fname = f'pct_chg_{freq}.csv'
    elif freq == 'd':
        fname = 'pct_chg.csv'
    else:
        raise RuntimeError("Unsupported return Type!")
        
    pct_chg = pd.read_csv(os.path.join(wdir, 'quote_data', fname),engine='c', index_col=[0], encoding='gbk')
    '''
    if freq=='6M':
        parm = 5
    elif freq=='M':
        parm = 0
    sql_zdf='''
     select a.s_info_windcode,
           to_date(a.trade_dt,'yyyymmdd')trade_dt,
           --a.S_MQ_PCTCHANGE,
           exp(sum(ln(1 + a.S_MQ_PCTCHANGE / 100))
               over(partition by a.s_info_windcode order by a.trade_dt
                    ROWS BETWEEN %s PRECEDING AND current row)) - 1 zdf
      from gfwind.AShareMonthlyYield a
     where a.trade_dt >= '20050101'
    '''%(parm)
    data = pd.read_sql(sql_zdf, engine)
    pct_chg = data.pivot(index='s_info_windcode',columns='trade_dt',values='zdf')
    
    pct_chg.columns = pd.to_datetime(pct_chg.columns)
    
    if freq.endswith('M'):
        dates = [date for date in weight.columns]
        pct_chg = pct_chg.loc[weight.index, dates]
        pct_chg.columns = weight.columns
        pct_chg = pct_chg.where(pd.notnull(wt), np.nan)
        pct_chg = pct_chg.dropna(how='all', axis=1)
    elif freq == 'd':
        pct_chg = pct_chg.loc[weight.index, :]
        start_date = f'{weight.columns[0].year}-{weight.columns[0].month}'
        end_date = f'{weight.columns[-1].year}-{weight.columns[-1].month}'
        pct_chg = pct_chg.loc[:, start_date:end_date]
    return pct_chg

def cal_group_ret(datdf):
    return (datdf['return'] * datdf['weight']).sum() / datdf['weight'].sum()

def cal_ind_ret_weight(weight, freq='6M',flag=True):
    dates = weight.columns.tolist()#所有日期，下跟权重
    ind_dat = get_ind_data(weight,flag) #index为基金，columns为日期 中间为所代表的行业
    ret_dat = get_stocks_ret(weight, freq)#index为基金，columns为日期，中间为收益率
        
    ind_return = []; ind_weight = []
    for date in dates: 
        if freq.endswith('M'):
            cur_stk = weight[date].dropna().index #去掉空行
            cur_ind = ind_dat.loc[cur_stk, date]
            cur_ret = ret_dat.loc[cur_stk, date]
            cur_weight = weight.loc[cur_stk, date]
                    
            cur_datdf = pd.concat([cur_ind, cur_ret, cur_weight], axis=1)
            cur_datdf.columns = ['industry', 'return', 'weight'] #简洁
            #各行业获利率/各行业权重
            cur_ind_ret = cur_datdf.groupby(['industry']).apply(cal_group_ret)
            #各行业权重
            cur_ind_weight = cur_datdf.groupby(['industry']).apply(lambda df: df['weight'].sum())
        cur_ind_ret.name = cur_ind_weight.name = date
        
        ind_return.append(cur_ind_ret)
        ind_weight.append(cur_ind_weight)
        
    ind_return = pd.DataFrame(ind_return).fillna(0) #空行填0
    ind_weight = pd.DataFrame(ind_weight).T.fillna(0)
    ind_weight /= ind_weight.sum()
    return ind_return, ind_weight.T  

def compute_bfg(ind_ret,select_ret,beyond_retp,beyond_retb,beyond_ret):#AR,SR,RP,RB,ER
    total=len(ind_ret.index)
    temp1=[];temp2=[];temp3=[]
    #zhh=list(ind_ret.index)
    for i in range(total):
        rsum=1
        rsum1=1
        rsum2=1
        for j in range(i):
            rsum*=(1+beyond_retp.iloc[j])
            rsum1*=(1+beyond_retp.iloc[j])
            rsum2*=(1+beyond_retp.iloc[j])
        rsum*=ind_ret.iloc[i]
        rsum1*=select_ret.iloc[i]
        rsum2*=beyond_ret.iloc[i]
        for j in range(i+1,total):
            rsum*=(1+beyond_retb.iloc[j])
            rsum1*=(1+beyond_retb.iloc[j])
            rsum2*=(1+beyond_retb.iloc[j])
        temp1.append(rsum)
        temp2.append(rsum1)
        temp3.append(rsum2)
    
    p_ind_ret=pd.DataFrame(temp1,index=ind_ret.index)
    p_select_ret=pd.DataFrame(temp2,index=ind_ret.index)
    p_beyond_ret=pd.DataFrame(temp3,index=ind_ret.index)
    
    #p_ind_ret.index=p_select_ret=p_beyond_ret.index=ind_ret.index
    res=pd.concat([p_ind_ret,p_select_ret,p_beyond_ret],axis=1)
    res.columns=["配置收益AR\'","选择收益SR\'","超额收益ER\'"]
    res1=res.copy()
    
    #DataFrame变形
    #翻转--索引变为列--拼接算法名字，总配置收益，总选股收益，总收益
    res=res.T
    res.reset_index(inplace=True)
    temp=pd.DataFrame(index=res.index)
    temp["算法名称"]=["GRAP算法",np.nan,np.nan]
    res["总和收益"]=res.sum(axis=1)#是否会把
    res["总配置收益"]=[res.iloc[0]["总和收益"],np.nan,np.nan]
    res["总选股收益"]=[res.iloc[1]["总和收益"],np.nan,np.nan]
    res["总超额收益"]=[res.iloc[2]["总和收益"],np.nan,np.nan]
    res.drop(['总和收益'],axis=1,inplace=True)#删除一列的做法
    res=pd.concat([temp,res],axis=1)
    res.to_excel(os.path.join(wdir,"temp","多期算法.xlsx"),encoding="gbk")
    #res1.to_excel(os.path.join(wdir,"temp","多期算法.xlsx"),encoding="gbk")
    return res1

def compute_other(dat):#2,3,4,5
    result={}
    
    resl=pd.DataFrame()#这个是字母l
    #名义复合法
    res=pd.DataFrame(index=dat.index,columns=["配置效应(AR\')","选股效应(SR\')","单期总收益(ER\')"])
    RB=RP=RA=1
    for date in dat.index:
        RB*=1+dat.loc[date,"re_b"]
        RP*=1+dat.loc[date,"re_p"]
        
    RB-=1
    RP-=1
    for i in range(0,len(AKH_wp.index)):
        RA*=(1+(AKH_wp.iloc[i]*AKH_rb.iloc[i]).sum())
    RA-=1
    RS=RP-RB-RA
    #能否这么去求
    zAR=RA
    zSR=RS
    
    res=res.T
    res.reset_index(inplace=True)
    temp=pd.DataFrame(index=res.index)
    temp["算法名称"]=["名义复合法",np.nan,np.nan]
    res["总和收益"]=res.sum(axis=1)#是否会把
    res["总配置收益"]=[res.iloc[0]["总和收益"],np.nan,np.nan]
    res["总选股收益"]=[res.iloc[1]["总和收益"],np.nan,np.nan]
    res["总超额收益"]=[res.iloc[2]["总和收益"],np.nan,np.nan]
    res.drop(['总和收益'],axis=1,inplace=True)#删除一列的做法
    res=pd.concat([temp,res],axis=1)
    res["总配置收益"]=[zAR,np.nan,np.nan]
    res["总选股收益"]=[zSR,np.nan,np.nan]
    res["总超额收益"]=[zAR+zSR,np.nan,np.nan]
    resl=pd.concat([resl,res])
    
    #AKH算法    
    res=pd.DataFrame(index=dat.index,columns=["配置效应(AR\')","选股效应(SR\')","单期总收益(ER\')"])
    for i in range(0,len(res.index)):
        if i==0:
            res.iloc[i]=[np.nan,np.nan,np.nan]
            continue
        #iloc中使用整数调用  sum中填axis也会出错
        Rtb=(1+dat.iloc[i-1,3])*(AKH_wb.iloc[i-1]*AKH_rb.iloc[i]).sum()#3-->re_b
        Rtp=(1+dat.iloc[i-1,2])*(AKH_wp.iloc[i-1]*AKH_rp.iloc[i]).sum()
        Rts=(1+dat.iloc[i-1,2])*(AKH_wb.iloc[i-1]*AKH_rp.iloc[i]).sum()
        Rta=(1+dat.iloc[i-1,3])*(AKH_wp.iloc[i-1]*AKH_rb.iloc[i]).sum()
        ARt=Rta-Rtb
        SRt=Rts-Rtb
        IRt=Rtp-Rtb-ARt-SRt
        SRt+=IRt
        res.iloc[i]=[ARt,SRt,Rtp-Rtb]
    result["AKH算法"]=res
    
    #Carino算法
    res=pd.DataFrame(index=dat.index,columns=["配置效应(AR\')","选股效应(SR\')","单期总收益(ER\')"])
    k=(math.log(1+RP)-math.log(1+RB))/(RP-RB)
    for date in dat.index:
        kt=(math.log(1+dat.loc[date,"re_p"])-math.log(1+dat.loc[date,"re_b"]))/(dat.loc[date,"re_p"]-dat.loc[date,"re_b"])
        a=kt/k*dat.loc[date,"配置效应(AR)"]
        b=kt/k*dat.loc[date,"选股效应(SR)"]
        c=a+b
        res.loc[date]=[a,b,c]
    result["Carino算法"]=res
    
    #Menchero算法
    res=pd.DataFrame(index=dat.index,columns=["配置效应(AR\')","选股效应(SR\')","单期总收益(ER\')"])
    T=len(dat.index)
    M=(RP-RB)/len(dat.index)/(pow(1+RP,1/T)-pow(1+RB,1/T))
    sum1=sum2=0
    for date in dat.index:
        sum1+=(dat.loc[date,"re_p"]-dat.loc[date,"re_b"])
        sum2+=pow(dat.loc[date,"re_p"]-dat.loc[date,"re_b"],2)
    for date in dat.index:
        at=(RP-RB-M*sum1)/sum2*(dat.loc[date,"re_p"]-dat.loc[date,"re_b"])
        a=(M+at)*dat.loc[date,"配置效应(AR)"]
        b=(M+at)*dat.loc[date,"选股效应(SR)"]
        c=a+b
        res.loc[date]=[a,b,c]
    result["Menchero算法"]=res
    
    #frongello算法
    res=pd.DataFrame(index=dat.index,columns=["配置效应(AR\')","选股效应(SR\')","单期总收益(ER\')"])
    for i in range(0,len(dat.index)):
        temp1=1
        temp2=0
        temp3=0
        for j in range(0,i):
            temp1*=1+dat.iloc[j,2]#re_p
            temp2+=res.iloc[j,0]#AR'
            temp3+=res.iloc[j,1]#SR'
            
        a=dat.iloc[i,0]*temp1+dat.iloc[i,3]*temp2
        b=dat.iloc[i,1]*temp1+dat.iloc[i,3]*temp3
        c=a+b
        res.iloc[i]=[a,b,c]
    result["frongello算法"]=res
    
    #耦合
    list1=["AKH算法","Carino算法","Menchero算法","frongello算法"]
    count=0
    for key in result.keys():
        res=result[key]
        res=res.T
        res["总和收益"]=res.sum(axis=1)#是否会把
        res.reset_index(inplace=True)
        temp=pd.DataFrame(index=res.index)
        temp["算法名称"]=[list1[count],np.nan,np.nan]
        res["总配置收益"]=[res.iloc[0]["总和收益"],np.nan,np.nan]
        res["总选股收益"]=[res.iloc[1]["总和收益"],np.nan,np.nan]
        res["总超额收益"]=[res.iloc[2]["总和收益"],np.nan,np.nan]
        res.drop(['总和收益'],axis=1,inplace=True)#删除一列的做法
        res=pd.concat([temp,res],axis=1)
        resl=pd.concat([resl,res])
        count+=1
    
    #读取GRAP结果
    temp=pd.read_excel(os.path.join(wdir,"temp","多期算法.xlsx"),encoding="gbk")
    resl=pd.concat([resl,temp])
    #全部输出到一个excel中
    resl.to_excel(os.path.join(wdir,"photo","多期","多期算法.xlsx"),encoding="gbk")
    return result
    
#堆叠图
def drawphotodui(brinson_stock):
    #brinson_stock[["配置效应(AR)","选股效应(SR)"]].plot(kind="bar",width=0.35,colormap="Accent_r",grid=True,stacked=True)
    
    #flag=["第一期","第二期","第三期","第四期","第五期"]
    flag=[str(elem)[:10] for elem in brinson_stock.index]
    x=[i*0.5 for i in range(0,len(flag))]
    
    
    y1=list(brinson_stock["配置效应(AR)"])
    y2=list(brinson_stock["选股效应(SR)"])
    plt.figure()
    #画条形图
    ret1=plt.bar(x,y1,color='g',width=0.3,label="配置效应(AR)",tick_label=flag,alpha=0.5)#,edgecolor='k'
    y11=[]#接在y1的点
    lenth=len(y1)
    for i in range(0,lenth):#堆叠图
        if y1[i]*y2[i]>0:
            y11.append(y1[i])
        else:
            y11.append(0)
    ret2=plt.bar(x,y2,color='gray',bottom=y11,width=0.3,label="选股效应(SR)")#,tick_label=flag,edgecolor='k'
    #标数字
    count=0
    for a,b in zip(ret1,ret2):
        plt.text(a.get_x()+a.get_width()/2,a.get_height()/2,"{:.2%}".format(a.get_height()),ha="center",fontsize=12)
        plt.text(b.get_x()+b.get_width()/2,y11[count]+b.get_height()/2,"{:.2%}".format(b.get_height()),ha="center",fontsize=12)
        count+=1  
    #坐标调整
    ax=plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position("zero")#设置坐标轴位置
    ax.xaxis.set_ticks_position("top")  #设置刻度的位置
    #ax.yaxis.set_ticks_position('left')
    plt.legend(loc="upper left")
    plt.grid(True,axis='y')
    plt.show()
    return
#分开画图
def drawphotofen(brinson_stock):
    fbrinson_stock=brinson_stock.copy()
    fbrinson_stock.index=[str(elem)[:10] for elem in brinson_stock.index]
    fbrinson_stock[["配置效应(AR)","选股效应(SR)"]].plot(kind="bar",width=0.35,colormap="Accent_r",stacked=False)
    ax=plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for a,b in zip(list(brinson_stock.index),list(brinson_stock["配置效应(AR)"])):
        plt.text(a,b,"hi")
    ax.spines["bottom"].set_position("zero")#设置坐标轴位置
    ax.xaxis.set_ticks_position("top")  #设置刻度的位置
    #ax.set_xticks(["第一期","第二期","第三期","第四期","第五期"])
    plt.legend(loc="left right")
    plt.grid(True,axis='y')
    
def brinson_attr_asset(stock_weight, asset_weight, fund_code, stock_bm='000300.SH', 
                       bond_bm='000012.SH', freq='6M', version=2, verbose=False,flag=True):
    #基金和沪深300基准的结果
    brinson_stock = brinson_attr_stock(stock_weight, asset_weight,stock_bm, freq, version, 
                                       verbose, fund_code,flag)
    
    brinson_stock.index = [get_trade_date(date,1) for date in brinson_stock.index]#调整日期
    
    AKH_wb.index=[get_trade_date(date,1) for date in AKH_wb.index]
    AKH_wp.index=[get_trade_date(date,1) for date in AKH_wp.index]
    AKH_rb.index=[get_trade_date(date,1) for date in AKH_rb.index]
    AKH_rp.index=[get_trade_date(date,1) for date in AKH_rp.index]
    
    #输出基金行业基准和行业权重和收益率
    AKH_wb.to_excel(os.path.join(wdir,"temp","wb.xlsx"),encoding="gbk")
    AKH_wp.to_excel(os.path.join(wdir,"temp","wp.xlsx"),encoding="gbk")
    AKH_rb.to_excel(os.path.join(wdir,"temp","rb.xlsx"),encoding="gbk")
    AKH_rp.to_excel(os.path.join(wdir,"temp","rp.xlsx"),encoding="gbk")
    
    #基金的实际收益率
    #fund_ret = get_index_ret(fund_code, freq)
    #fund_ret = fund_ret.loc[brinson_stock.index]
    
    #drawphotodui(brinson_stock)
    #BF+GRAP
    ind_ret = brinson_stock['配置效应(AR)']
    select_ret = brinson_stock['选股效应(SR)']
    beyond_retp=brinson_stock["re_p"]
    beyond_retb=brinson_stock["re_b"]
    beyond_ret=beyond_retp-beyond_retb
    bfg=compute_bfg(ind_ret,select_ret,beyond_retp,beyond_retb,beyond_ret)
    result=compute_other(brinson_stock)
    res=pd.concat([brinson_stock,bfg],axis=1)
    res.loc["总计"]=bfg.sum(axis=0)
    res.loc["总计","配置效应(AR)"]=np.nan  
    res.loc["总计","选股效应(SR)"]=np.nan
    res.loc["总计","re_p"]=np.nan
    res.loc["总计","re_b"]=np.nan
    res["误差"]=res["re_p"]-res["re_b"]-res["配置效应(AR)"]-res["选股效应(SR)"]
    return res

def load_index_weight(benchmark='000300.SH',flag=True):
    filepath=os.path.join(wdir,"temp")
    if flag==False:
        bm_weight=pd.read_excel(os.path.join(filepath,"bm_weight.xlsx"))
        bm_weight=bm_weight.set_index(["s_con_windcode"])
        return bm_weight
    sql_weight = '''
     select a.s_con_windcode,
     to_date(a.trade_dt,'yyyymmdd')trade_dt,
     a.i_weight from
     gfwind.AIndexHS300FreeWeight a
     inner join (
     select max(a.trade_days) trade_days
       from gfwind.AShareCalendar a
      where a.s_info_exchmarket = 'SSE'
        and a.trade_days >= '20090123'
        and a.trade_days <= to_char(sysdate, 'yyyymmdd')
      group by trunc(to_date(TRADE_DAYS, 'yyyymmdd'), 'mm')
     )b
     on a.trade_dt = b.trade_days
     where a.S_INFO_WINDCODE = '%s'
    '''%(benchmark)
    data = pd.read_sql(sql_weight, engine)
    #index=基金代码 columns=贸易日期 中间为weight
    bm_weight=data.pivot(index='s_con_windcode',columns='trade_dt',values='i_weight')
    bm_weight=bm_weight.dropna(how='all')#行中全为Nan
    if flag==True:
        bm_weight.to_excel(os.path.join(filepath,"bm_weight.xlsx"),encoding="gbk")
    return bm_weight
#股票部分和沪深300做一个比较
def brinson_attr_stock(weight, asset_weight,benchmark='000300.SH', freq='6M', version=2, 
                       verbose=False, fund_code=None,flag=True):
    #计算所有的单期归因结果
    r1 = ['配置效应(AR)','选股效应(SR)','交互效应(IR)']  #BHB
    r2 = ['配置效应(AR)','选股效应(SR)']  #BF
    
    #wB index为股票编号 columns为日期
    #bm_weight = pd.read_csv(os.path.join(wdir, 'index_weight', f'{benchmark.split(".")[0]}.csv'),
                           # engine='python', encoding='gbk', index_col=[0])
    bm_weight = load_index_weight(benchmark,flag)
    bm_weight.columns = pd.to_datetime(bm_weight.columns)
    bm_weight = bm_weight[weight.columns] #提取投资组合日期的权重
    bm_weight /= bm_weight.sum()  #重新分配比例
    #wP,rP,wB,rB index为日期 columns为行业
    pt_ind_ret, pt_ind_weight = cal_ind_ret_weight(weight,flag=flag)
    bm_ind_ret, bm_ind_weight = cal_ind_ret_weight(bm_weight,flag=flag)
    bm_ind_weight.iloc[0]#第一行
    #将权重加入到WP中
    """
    temp=asset_weight.index
    asset_weight.index=pt_ind_weight.index
    for c in pt_ind_weight.index:
        pt_ind_weight.loc[c]=pt_ind_weight.loc[c]*asset_weight.loc[c,"pt_stock"]
    asset_weight.index=temp
    """
    mut_ind = pt_ind_weight.columns | bm_ind_weight.columns#行业做一个并集操作
    #在pt_ind_weight中补充缺少的行业
    if len(mut_ind.difference(pt_ind_weight.columns)) > 0:#mut_ind-pt_ind_weight.columns 即基准行业中有的，投资行业中没有
        cols = mut_ind.difference(pt_ind_weight.columns)#单独拎出来独有的基准行业
        for col in cols:
            pt_ind_weight.loc[:, col] = 0
            pt_ind_ret.loc[:, col] = 0
    #在bm_ind_weight中补充缺少的行业       
    if len(mut_ind.difference(bm_ind_weight.columns)) > 0:
        cols = mut_ind.difference(bm_ind_weight.columns)
        for col in cols:
            bm_ind_weight.loc[:, col] = 0
            bm_ind_ret.loc[:, col] = 0
    #index为行业 columns为AR，SR等
    brinson_single = brinson_attr_single_period(pt_ind_ret, pt_ind_weight, 
                                    bm_ind_ret, bm_ind_weight, version)
    
    #保留单期的结果 如果文件已经存在的话，不会写入到文件中
    if verbose:
        if fund_code is None:
            raise RuntimeError('Need to pass in "fund_code" to save attribution result file!')
        filepath=os.path.join(wdir,"brinson_result",f"{fund_code}_res.xlsx")
        excel_writer=pd.ExcelWriter(filepath)
        for dkey in brinson_single.keys():
            temp=str(get_trade_date(pd.to_datetime(dkey),1))[:10]
            brinson_single[dkey].to_excel(excel_writer,sheet_name=temp,encoding="gbk") 
        excel_writer.save()
        excel_writer.close() 
    
    brinson_returns = r1 if version == 1 else r2
    single_ret = pd.DataFrame()#index为日期 columns为配置，选择收益
    for r in brinson_returns:
        dat_panel=pd.DataFrame()
        for date in brinson_single.keys():
            dat_panel=pd.concat([dat_panel,brinson_single[date][r]],axis=1)#横向连接
        if"总计" in dat_panel.index:
            dat_panel.drop(["总计"],inplace=True)
        dat_panel.columns=brinson_single.keys()
        single_ret[r]=dat_panel.sum(axis=0)
   
    #每个单期内的投资收益和基准收益
    port_ret = (pt_ind_ret * pt_ind_weight).sum(axis=1)#对应相乘在横向累和 单期RP
    bm_ret = (bm_ind_ret * bm_ind_weight).sum(axis=1)#单期RB
    
    single_ret.index = pd.to_datetime(single_ret.index)#转变一下格式
    #横向拼接
    single_ret = pd.concat([single_ret, port_ret, bm_ret], axis=1)
    single_ret.columns = brinson_returns + ['re_p', 're_b']#赋予列标
    return single_ret#index为日期

def brinson_attr_single_period(pt_ret, pt_weight, bm_ret, bm_weight, version=2): #rP,wP,rB,wB BF 当成一个一维数组，计算单期的收益分解
    result = {}
    global AKH_wb,AKH_wp,AKH_rb,AKH_rp
    AKH_wp=pt_weight.copy(deep=True)
    AKH_rp=pt_ret.copy(deep=True)
    AKH_rb=bm_ret.copy(deep=True)
    AKH_wb=bm_weight.copy(deep=True)
    bm_total_ret = (bm_ret * bm_weight).sum(axis=1)  #基准总收益 RB 行累和
    flag=False#只抓取一期
    for date in pt_ret.index: #行的标志  遍历日期
        res = pd.DataFrame(index=bm_weight.columns)#index为行业
        res['组合权重'] = pt_weight.loc[date]  #返回一行，type为series
        res['基准权重'] = bm_weight.loc[date]
        res['组合收益'] = pt_ret.loc[date]
        res['基准收益'] = bm_ret.loc[date]
        if version == 1: #BHB
            res['配置效应(AR)'] = (res['组合权重'] - res['基准权重']) * res['基准收益']
            res['选股效应(SR)'] = res['基准权重'] * (res['组合收益'] - res['基准收益'])
            res['交互效应(IR)'] = (res['组合权重'] - res['基准权重']) * (res['组合收益'] - res['基准收益'])
        elif version == 2: #BF
            res['配置效应(AR)'] = (res['组合权重'] - res['基准权重']) * (res['基准收益'] - bm_total_ret.loc[date])
            res['选股效应(SR)'] = res['组合权重'] * (res['组合收益'] - res['基准收益'])
        res['超额收益'] = res['组合权重'] * res['组合收益'] - res['基准收益'] * res['基准权重']
        res.loc['总计'] = res.sum()  #列的累和 
        res.loc['总计', ['组合收益', '基准收益']] = np.nan  #不累和组合收益和基准收益
        result[str(date)[:10]] = res  #date的前10个数为key 每一个日期的结果
        #需要的图表
        if flag==False:
            flag=True
            single_res=pd.DataFrame(index=res.index)
            #single_res["行业"]=res.index
            single_res["基准权重wB"]=res["基准权重"]
            single_res["组合权重wP"]=res["组合权重"]
            single_res["主动权重(wP-wB)"]=res["组合权重"]-res["基准权重"]
            single_res["基准收益rBi"]=res["基准收益"]
            single_res["组合收益rPi"]=res["组合收益"]
            single_res["主动收益(rP-rB)"]=res["组合收益"]-res["基准权重"]
            single_res["配置收益(AR)"]=res["配置效应(AR)"]
            single_res["选股收益(SR)"]=res["选股效应(SR)"]
            single_res["总超额收益"]=res["配置效应(AR)"]+res["选股效应(SR)"]
            filepath1=os.path.join(wdir,"brinson_result","单期全.xlsx")
            single_res.to_excel(filepath1,encoding="gbk")
        
    #result = pd.Panel(result)
    return result

def get_cdate(date):#就是调整到月末最后一天
    nextdate = date + pd.tseries.offsets.MonthEnd(1)#nextdate为月末最后一天
    if nextdate.month > date.month:
        cdate = date
    else:
        cdate = nextdate
    return cdate

def get_index_ret(code, freq='6M'):
#将日收益率转换为设定频率的收益率。
#例如，freq默认为6个月时，将日收益率转换为半年度收益，且默认计算时间范围为
#1-6月及7-12月，计算起始日期选择基金或者指数成立后的首个完整的半年度的首个
#月份第1个交易日（1月或者7月）

    #ret = pd.read_csv(os.path.join(wdir, 'quote_data', f'{code}.csv'), parse_dates=True, engine='python', encoding='gbk', index_col=[0])
    sql_index = '''
    select to_date(a.trade_dt,'yyyymmdd')trade_dt,a.s_dq_pctchange/100 pct_change from gfwind.AIndexEODPrices a
    where a.s_info_windcode = '%s'
    and a.trade_dt >='20050101'
    union all
    select to_date(a.trade_dt,'yyyymmdd')trade_dt,a.S_DQ_PCTCHANGE/100 pct_change from gfwind.CBIndexEODPrices a
    where a.s_info_windcode = '%s'
    and a.trade_dt >='20050101'
    union all
    select to_date(a.trade_dt,'yyyymmdd')trade_dt,a.F_AVGRETURN_DAY/100 from 
    gfwind.ChinaMFPerformance a
    where a.s_info_windcode = '%s' 
    and a.trade_dt >= '20050101'
    and a.F_AVGRETURN_DAY is not null
    '''%(code,code,code)
    data = pd.read_sql(sql_index, engine)
    ret = data.set_index('trade_dt')
    
    ret = ret.dropna(how='any', axis=0)['pct_change']
    if freq.endswith('M') and freq != 'M':
        num_months = int(freq[:-1])
        freq = 'M'
    else:
        num_months = 0
    ret = ret.groupby(pd.Grouper(freq=freq)).apply(lambda df: ((1+df).cumprod()-1).iloc[-1]).iloc[1:]
    if num_months > 0:
        if ret.index[0].month % num_months != 0:
            startdate = list(dropwhile(lambda d: d.month % num_months != 0, ret.index))[0]
            ret = ret.loc[startdate:]
        if ret.index[-1].month % num_months != 0:
            enddate = list(dropwhile(lambda d: d.month % num_months != 0, ret.index[::-1]))[0]
            ret = ret.loc[:enddate]
        ret = ret.groupby(pd.Grouper(freq=f'{num_months}M')).apply(lambda df: ((1+df).cumprod()-1).iloc[-1]).iloc[1:]
    return ret

def clean_index_quote(save_cols=('close',), save_ori=False):
    """
       input: 
	   从wind终端下载的基金/指数日频行情数据文件，文件名格式：'基金代码'.xls
       output:
           根据close_price计算日收益率, 根据save_cols参数决定所要原始的数据列；
	   结果存储为csv，通过save_ori关键字参数决定是否保留原始xls文件，默认值为False
	   存储结果见quote_data文件夹
    """		    
    quote_dir = os.path.join(wdir, 'quote_data')
    files = [f for f in os.listdir(quote_dir) if f.endswith('xls')]
    col_map = {
            'open': '开盘价(元)',
            'close': '收盘价(元)',
            'high': '最高价(元)',
            'low': '最低价(元)',
            'name': '名称',
            'code': '代码',
            'amount': '成交额(百万)',
            'volume': '成交量(股)',
            }
    
    save_map = {col_map[col.lower()]:col.lower() for col in save_cols 
                if col.lower() in col_map.keys()}
    
    for f in files: 
        dat = pd.read_excel(os.path.join(quote_dir, f), encoding='gbk')
        dat = dat.dropna(how='any', axis=0)#行有nan的情况下删除
        dat['pct_change'] = dat[['收盘价(元)']].pct_change()
        dat = dat.rename(columns=save_map)
        dat = dat.set_index(['日期'])
        dat.index.name = 'date'
        dat = dat[list(save_map.values()) + ['pct_change']]
        dat.iloc[1:].to_csv(os.path.join(quote_dir, f[:-4]+'.csv'))
        if not save_ori:
            os.remove(os.path.join(quote_dir, f))

def clean_fund_holding(save_ori=True):
    """
       input: 
	   从wind终端下载的基金持仓明细文件，文件名格式：'基金代码'持股.csv
       output:
	   结果存储为xlsx，每个sheet名为对应持仓报告期日期，
	   通过save_ori关键字参数决定是否保留原始csv文件，默认值为True
	   存储结果见fund_holding文件夹
    """
    fund_dir = os.path.join(wdir, 'fund_holding')
    files = [f for f in os.listdir(fund_dir) if '持股' in f]
    for f in files: 
        try:
            dat = pd.read_csv(os.path.join(fund_dir, f), engine='python', 
                              encoding='utf-8', index_col=[0], parse_dates=True)
        except UnicodeDecodeError:
            dat = pd.read_csv(os.path.join(fund_dir, f), engine='python', 
                              encoding='gbk', index_col=[0], parse_dates=True)
  
        dat = dat.reset_index()
        dat['报告期'] = dat['报告期'].map(lambda d: str(d)[:10])
        del dat['序号']
        dat = dat.set_index(['品种代码', '报告期'])
        dat = dat.to_panel()
        dat = dat.swapaxes(0, 2)
        dat.to_excel(os.path.join(fund_dir, f.split('持股')[0]+'.xlsx'),
                     encoding='gbk')
        if not save_ori:
            os.remove(os.path.join(fund_dir, f))

def read_fund_holding(code, index=None, bm_stock_wt=0.80):
    fund_dir = os.path.join(wdir, 'fund_holding')
    #read出错没有拿到所有的sheet
    dat = pd.read_excel(os.path.join(fund_dir, code+'.xlsx'), encoding='gbk', sheet_name=None)
    if index:
        index_weight = pd.read_csv(os.path.join(wdir, 'index_weight', f'{index}.csv'),
                         engine='python', encoding='gbk', index_col=[0])
        index_weight.columns = pd.to_datetime(index_weight.columns)
        
    stock_weight = pd.DataFrame(); asset_weight = pd.DataFrame(columns=['stock', 'bond'])
    for date in dat.keys(): #dat.keys()能够得到sheet的名字
        panel = dat[date]   #得到一个DataFrame
        date = pd.to_datetime(date)
        panel = panel[panel['所属行业名称'].notnull()] #产生一个表 notnull---->>True 就是去掉空行
        panel = panel.rename(columns={'占股票市值比(%)': 'weight',
                                      '品种代码':'code'})
        if index:
            tdate = get_trade_date(date)
            panel = panel[panel['code'].isin(index_weight[tdate].dropna().index)]
        panel = panel.set_index(['code']) #code这个列没了，作为了dataframe的行索引
        stock_weight = pd.concat([stock_weight, panel['weight']], axis=1)#行拼接 没有/100
        asset_weight.loc[date] = [panel['占基金净值比(%)'].sum()/100, 1 - panel['占基金净值比(%)'].sum()/100]
    
    asset_weight.columns = ['pt_stock', 'pt_bond']
    asset_weight['bm_stock'] = bm_stock_wt
    asset_weight['bm_bond'] = 1 - bm_stock_wt
    stock_weight.columns=[pd.to_datetime(date) for date in dat.keys()]
    stock_weight.columns = [get_trade_date(pd.to_datetime(date)) for date in dat.keys()]
    return stock_weight, asset_weight
def load_data_weight(fund_code,bm_stock_wt,flag=True):
    filepath=os.path.join(wdir,"temp")
    if flag==False:
        stock_weight=pd.read_excel(os.path.join(filepath,"stock_weight.xlsx"))
        asset_weight=pd.read_excel(os.path.join(filepath,"asset_weight.xlsx"))
        stock_weight=stock_weight.set_index(["s_info_stockwindcode"])
        asset_weight=asset_weight.set_index(["f_prt_enddate"])
        return stock_weight,asset_weight
        
    fund_code=fund_code.replace('.OF','')
    fund_code="".join(filter(str.isdigit, fund_code))
    print('fundcode data is load: ',fund_code)
    sql_weight = '''
      select to_date(f_prt_enddate,'yyyymmdd')f_prt_enddate,
      a.S_INFO_STOCKWINDCODE,
      a.STOCK_PER,
      a.F_PRT_STKVALUETONAV
      from gfwind.ChinaMutualFundStockPortfolio a
     where substr(a.s_info_windcode,1,6) = '%s'
     and a.f_prt_enddate in ('20200630','20201231','20210630')
    '''%(fund_code)
    #'20170630','20171231','20180630','20181231','20190630','20191231',,'20211231'
    data = pd.read_sql(sql_weight, engine)
    #index=股票编码，column=date ，比例
    stock_weight = data[['s_info_stockwindcode','f_prt_enddate','stock_per']].pivot(index='s_info_stockwindcode',columns='f_prt_enddate',values='stock_per')
    asset_weight=data[['f_prt_enddate','f_prt_stkvaluetonav']].groupby(['f_prt_enddate']).sum()/100
    asset_weight.columns = ['pt_stock']
    asset_weight['pt_bond'] = 1 - asset_weight['pt_stock']
    asset_weight['bm_stock'] = bm_stock_wt
    asset_weight['bm_bond'] = 1 - bm_stock_wt
    
    stock_weight.columns = [get_trade_date(pd.to_datetime(date)) for date in stock_weight.columns]
    if flag==True:
        stock_weight.to_excel(os.path.join(filepath,"stock_weight.xlsx"),encoding="gbk")
        asset_weight.to_excel(os.path.join(filepath,"asset_weight.xlsx"),encoding="gbk")
    return stock_weight,asset_weight
def brinson_attribution():
    starttime=time.time()
    fund_code = '519702.OF'        #基金代码
    bond_benchmark = '000012.SH'   #基金对应基准债券指数代码
    stock_benchmark = '000300.SH'  #基金对应基准股票指数基准代码
    bm_stock_wt = 0.80             #基准股票指数比例
    version = 2              #brinson归因模型版本 1--BHB; 2--BF
    freq = '6M'               #归因频率，与所选基金持仓频率对应，默认选择基金的半年报和年报
    verbose = True        #是否存储单层brinson归因结果（股票行业配置和选股效应）
    flag=True          #是否需要从数据库中导入数据
    print("version: %d" %version)
    #clean_index_quote()    #清洗并计算基金/基准指数日收益率
    #clean_fund_holding()   #清洗基金持仓文件
    #stock_weight, asset_weight = read_fund_holding(fund_code, None, bm_stock_wt) 
    stock_weight, asset_weight = load_data_weight(fund_code,bm_stock_wt,flag)
    res=brinson_attr_asset(stock_weight, asset_weight, fund_code, stock_benchmark, 
                             bond_benchmark, freq, version, verbose,flag)
    if not os.path.exists(os.path.join(wdir, 'brinson_result')):
        os.mkdir(os.path.join(wdir, 'brinson_result'))
    #含有BF-GRAP的单期
    res.to_csv(os.path.join(wdir, 'brinson_result', f'{fund_code}.csv'), encoding='gbk')
    print(f'Finish for {fund_code}.')
    endtime=time.time()
    dtime=endtime-starttime
    print(f"程序运行时间为{dtime}")
    
if __name__ == '__main__':
    brinson_attribution()
    
     
    
