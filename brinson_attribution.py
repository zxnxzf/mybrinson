# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 08:42:47 2019

@author: HP
"""
#师兄改的，只含有输入输出
import os
import numpy as np
import pandas as pd
from itertools import dropwhile
import warnings
warnings.filterwarnings('ignore') 

wdir = os.path.dirname(__file__)
import time
import cx_Oracle
from sqlalchemy import create_engine
conn_string='oracle+cx_oracle://hurz:Gfhurz123@10.88.102.82:1521/?service_name=FINCHINA'
engine = create_engine(conn_string, echo=False)
#传入自然日获取月末最后一个工作日
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

def get_trade_date_list():
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
    data = data.set_index('zrr')
    
    return data['ym'].tolist()
def load_ind_data(ind_type='zx'):
    #date_list = [20220531,20220630]
    date_list = get_trade_date_list()
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
        
        data = pd.read_sql(sql_data, engine)
        ind_data=data.pivot(index='s_info_windcode',columns='sk_date',values='industriesname')
        ind_panel = pd.concat([ind_panel, ind_data], axis=1)
    return ind_panel

def get_ind_data(weight, ind_type='zx'):
    #industry_dat = pd.read_csv(os.path.join(wdir, 'quote_data', f'industry_{ind_type}.csv'),encoding='gbk', engine='python', index_col=[0])
    industry_dat = load_ind_data()
    industry_dat.columns = pd.to_datetime(industry_dat.columns)
    industry_dat = industry_dat.loc[weight.index, weight.columns]
    industry_dat = industry_dat.where(pd.notnull(weight), np.nan)
    return industry_dat



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

def cal_ind_ret_weight(weight, freq='6M'):
    dates = weight.columns.tolist()
    ind_dat = get_ind_data(weight)
    ret_dat = get_stocks_ret(weight, freq)
        
    ind_return = []
    ind_weight = []
    for date in dates: 
        if freq.endswith('M'):
            cur_stk = weight[date].dropna().index
            cur_ind = ind_dat.loc[cur_stk, date]
            cur_ret = ret_dat.loc[cur_stk, date]
            cur_weight = weight.loc[cur_stk, date]
                    
            cur_datdf = pd.concat([cur_ind, cur_ret, cur_weight], axis=1)
            cur_datdf.columns = ['industry', 'return', 'weight']
            
            cur_ind_ret = cur_datdf.groupby(['industry']).apply(cal_group_ret)
            cur_ind_weight = cur_datdf.groupby(['industry']).apply(lambda df: df['weight'].sum())
        cur_ind_ret.name = cur_ind_weight.name = date
        
        ind_return.append(cur_ind_ret)
        ind_weight.append(cur_ind_weight)
        
    ind_return = pd.DataFrame(ind_return).fillna(0)
    ind_weight = pd.DataFrame(ind_weight).T.fillna(0)
    ind_weight /= ind_weight.sum()
    return ind_return, ind_weight.T    

def brinson_attr_asset(stock_weight, asset_weight, fund_code, stock_bm='000300.SH', 
                       bond_bm='000012.SH', freq='6M', version=2, verbose=False):
    brinson_stock = brinson_attr_stock(stock_weight, stock_bm, freq, version, 
                                       verbose, fund_code)
    brinson_stock.index = [get_trade_date(date,to_type=1) for date in brinson_stock.index]
    
    bond_bm_ret = get_index_ret(bond_bm, freq)
    bond_bm_ret = bond_bm_ret.loc[brinson_stock.index]
    
    stock_bm_ret = get_index_ret(stock_bm, freq)
    stock_bm_ret = stock_bm_ret.loc[brinson_stock.index]
    
#    bm_ret = brinson_stock['re_b'] * asset_weight['bm_stock'] + bond_bm_ret * asset_weight['bm_bond']
    bm_ret = stock_bm_ret * asset_weight['bm_stock'] + bond_bm_ret * asset_weight['bm_bond']
    
    fund_ret = get_index_ret(fund_code, freq)
    fund_ret = fund_ret.loc[brinson_stock.index]
    
    timing_ret = (asset_weight['pt_stock'] - asset_weight['bm_stock']) * (brinson_stock['re_b'] - bm_ret) + \
                 (asset_weight['pt_bond'] - asset_weight['bm_bond']) * (bond_bm_ret - bm_ret)
    ind_ret = brinson_stock['配置效应(AR)'] * asset_weight['pt_stock']
    select_ret = brinson_stock['选股效应(SR)'] * asset_weight['pt_stock']
    
    res_con_timing = pd.concat([fund_ret, bm_ret, timing_ret, ind_ret, select_ret], axis=1)
    res_con_timing.columns = ['基金收益', '基准实际收益', '大类资产择时收益(TR)', '配置效应(AR)', '选股效应(SR)']
    res_con_timing['估计误差'] = res_con_timing['基金收益'] - res_con_timing[['基准实际收益', '大类资产择时收益(TR)', '配置效应(AR)', '选股效应(SR)']].sum(axis=1)
    res_con_timing['是否调整'] = '调整后'
    
    res_wo_timing = pd.concat([fund_ret, bm_ret, brinson_stock[['配置效应(AR)', '选股效应(SR)']]], axis=1)
    res_wo_timing.columns = ['基金收益', '基准实际收益', '配置效应(AR)', '选股效应(SR)']
    res_wo_timing['大类资产择时收益(TR)'] = np.nan
    res_wo_timing['估计误差'] = res_wo_timing['基金收益'] - res_wo_timing[['基准实际收益', '配置效应(AR)', '选股效应(SR)']].sum(axis=1)
    res_wo_timing['是否调整'] = '调整前'

    res = pd.concat([res_con_timing, res_wo_timing], axis=0)
    res.index.name = 'date'
    res = res.reset_index()
    res = res.set_index(['date','是否调整'])
    res = res.sort_index()
    res = res[['基金收益', '基准实际收益', '大类资产择时收益(TR)',
               '选股效应(SR)', '配置效应(AR)', '估计误差']]
    return res

def load_index_weight(benchmark='000300.SH'):
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
    return bm_weight
    
def brinson_attr_stock(weight, benchmark='000300.SH', freq='6M', version=2, 
                       verbose=False, fund_code=None):
    r1 = ['配置效应(AR)','选股效应(SR)','交互效应(IR)']
    r2 = ['配置效应(AR)','选股效应(SR)']

    #bm_weight = pd.read_csv(os.path.join(wdir, 'index_weight', f'{benchmark.split(".")[0]}.csv'),engine='python', encoding='gbk', index_col=[0])
    bm_weight = load_index_weight(benchmark)
    bm_weight.columns = pd.to_datetime(bm_weight.columns)
    bm_weight = bm_weight[weight.columns]
    bm_weight=bm_weight.dropna(how='all')
    bm_weight /= bm_weight.sum()
        
    pt_ind_ret, pt_ind_weight = cal_ind_ret_weight(weight)
    bm_ind_ret, bm_ind_weight = cal_ind_ret_weight(bm_weight)
    #bm_ind_weight.iloc[0]

    mut_ind = pt_ind_weight.columns | bm_ind_weight.columns
    if len(mut_ind.difference(pt_ind_weight.columns)) > 0:
        cols = mut_ind.difference(pt_ind_weight.columns)
        for col in cols:
            pt_ind_weight.loc[:, col] = 0
            pt_ind_ret.loc[:, col] = 0
            
    if len(mut_ind.difference(bm_ind_weight.columns)) > 0:
        cols = mut_ind.difference(bm_ind_weight.columns)
        for col in cols:
            bm_ind_weight.loc[:, col] = 0
            bm_ind_ret.loc[:, col] = 0
    
    brinson_single = brinson_attr_single_period(pt_ind_ret, pt_ind_weight, 
                                    bm_ind_ret, bm_ind_weight, version)
    if verbose:
        if fund_code is None:
            raise RuntimeError('Need to pass in "fund_code" to save attribution result file!')
        #brinson_single.to_excel(os.path.join(wdir, 'brinson_result',f'{fund_code}_res.xlsx'), encoding='gbk')
        write = pd.ExcelWriter(os.path.join(wdir, 'brinson_result_test',f'{fund_code}_res.xlsx'))
        for key in brinson_single.keys():
            brinson_single[key].to_excel(excel_writer=write, encoding='gbk',sheet_name=key)
        write.save()
        write.close()
    brinson_returns = r1 if version == 1 else r2
    single_ret = pd.DataFrame()
    
    for r in brinson_returns:
        dat_panel = pd.DataFrame()
        for key in brinson_single.keys():
            df = brinson_single[key]
            if '总计' in df.index:
                df.drop(['总计'], inplace=True)
            dat_panel=pd.concat([dat_panel,df[r]],axis=1)
        dat_panel=dat_panel.sum(axis=0)
        dat_panel.index = pt_ind_ret.index
        single_ret[r] = dat_panel
    
    port_ret = (pt_ind_ret * pt_ind_weight).sum(axis=1)
    bm_ret = (bm_ind_ret * bm_ind_weight).sum(axis=1)
    
    single_ret.index = pd.to_datetime(single_ret.index)
    single_ret = pd.concat([single_ret, port_ret, bm_ret], axis=1)
    single_ret.columns = brinson_returns + ['re_p', 're_b']
    return single_ret

def brinson_attr_single_period(pt_ret, pt_weight, bm_ret, bm_weight, version=2):
    result = {}
    bm_total_ret = (bm_ret * bm_weight).sum(axis=1)
    for date in pt_ret.index: 
        res = pd.DataFrame(index=bm_weight.columns)
        res['组合权重'] = pt_weight.loc[date]
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
        res.loc['总计'] = res.sum()
        res.loc['总计', ['组合收益', '基准收益']] = np.nan
        result[str(date)[:10]] = res
    #result = pd.Panel(result)
    return result

def get_cdate(date):
    nextdate = date + pd.tseries.offsets.MonthEnd(1)
    if nextdate.month > date.month:
        cdate = date
    else:
        cdate = nextdate
    return cdate

def get_index_ret(code, freq='6M'):
	
    #ret = pd.read_csv(os.path.join(wdir, 'quote_data', f'{code}.csv'), parse_dates=True,engine='python', encoding='gbk', index_col=[0])
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

def read_fund_holding(code, index=None, bm_stock_wt=0.80):
    fund_dir = os.path.join(wdir, 'fund_holding')
    dat = pd.read_excel(os.path.join(fund_dir, code+'.xlsx'), sheet_name=None)
    if index:
        index_weight = pd.read_csv(os.path.join(wdir, 'index_weight', f'{index}.csv'),
                         engine='python', index_col=[0])
        index_weight.columns = pd.to_datetime(index_weight.columns)
        
    stock_weight = pd.DataFrame()
    asset_weight = pd.DataFrame(columns=['stock', 'bond'])
    for date in dat.keys(): 
        panel = dat[date]
        date = pd.to_datetime(date)
        panel = panel[panel['所属行业名称'].notnull()]
        panel = panel.rename(columns={'占股票市值比(%)': 'weight',
                                      '品种代码':'code'})
        if index:
            tdate = get_trade_date(date)
            panel = panel[panel['code'].isin(index_weight[tdate].dropna().index)]
        panel = panel.set_index(['code'])
        stock_weight = pd.concat([stock_weight, panel['weight']], axis=1)
        asset_weight.loc[date] = [panel['占基金净值比(%)'].sum()/100, 1 - panel['占基金净值比(%)'].sum()/100]
    
    asset_weight.columns = ['pt_stock', 'pt_bond']
    asset_weight['bm_stock'] = bm_stock_wt
    asset_weight['bm_bond'] = 1 - bm_stock_wt
    stock_weight.columns = [get_trade_date(pd.to_datetime(date)) for date in dat.keys()]
    return stock_weight, asset_weight

def brinson_attribution():
    fund_code = '540006.OF'        #基金代码
    bond_benchmark = '000012.SH'   #基金对应基准债券指数代码
    stock_benchmark = '000300.SH'  #基金对应基准股票指数基准代码
    bm_stock_wt = 0.80             #基准股票指数比例
    version = 2               #brinson归因模型版本 1--BHB; 2--BF
    freq = '6M'               #归因频率，与所选基金持仓频率对应，默认选择基金的半年报和年报
    verbose = True        #是否存储单层brinson归因结果（股票行业配置和选股效应）
    
    #clean_index_quote()    #清洗并计算基金/基准指数日收益率
    #clean_fund_holding()   #清洗基金持仓文件
    stock_weight, asset_weight = read_fund_holding(fund_code, None, bm_stock_wt) 
    res = brinson_attr_asset(stock_weight, asset_weight, fund_code, stock_benchmark, 
                             bond_benchmark, freq, version, verbose)
			    
    if not os.path.exists(os.path.join(wdir, 'brinson_result')):
        os.mkdir(os.path.join(wdir, 'brinson_result'))
    res.to_csv(os.path.join(wdir, 'brinson_result', f'{fund_code}.csv'), encoding='gbk')
    print(f'Finish for {fund_code}.')

def load_data_weight(fund_code,bm_stock_wt):
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
     and a.f_prt_enddate in ('20170630','20171231','20180630','20181231','20190630','20191231','20200630','20201231','20210630','20211231')
    '''%(fund_code)
    #'20190930','20200331','20200930'  '20170630','20171231','20180630','20181231','20190630','20191231','20200630','20201231','20210630','20211231'
    data = pd.read_sql(sql_weight, engine)
    #index=股票编码，column=date ，比例
    stock_weight = data[['s_info_stockwindcode','f_prt_enddate','stock_per']].pivot(index='s_info_stockwindcode',columns='f_prt_enddate',values='stock_per')
    asset_weight=data[['f_prt_enddate','f_prt_stkvaluetonav']].groupby(['f_prt_enddate']).sum()/100
    asset_weight.columns = ['pt_stock']
    asset_weight['pt_bond'] = 1 - asset_weight['pt_stock']
    asset_weight['bm_stock'] = bm_stock_wt
    asset_weight['bm_bond'] = 1 - bm_stock_wt
    
    stock_weight.columns = [get_trade_date(pd.to_datetime(date)) for date in stock_weight.columns]
    return stock_weight,asset_weight
if __name__ == '__main__':
    #brinson_attribution()
    #clean_index_quote()
    #clean_fund_holding()
    starttime = time.time()
    fund_code = '270028.OF'        #基金代码
    bond_benchmark = '000012.SH'   #基金对应基准债券指数代码
    stock_benchmark = '000300.SH'  #基金对应基准股票指数基准代码
    bm_stock_wt = 0.80             #基准股票指数比例
    version = 1              #brinson归因模型版本 1--BHB; 2--BF
    freq = '6M'               #归因频率，与所选基金持仓频率对应，默认选择基金的半年报和年报
    verbose = True        #是否存储单层brinson归因结果（股票行业配置和选股效应）
    #stock_weight, asset_weight = read_fund_holding(fund_code, None, bm_stock_wt) 
    stock_weight, asset_weight = load_data_weight(fund_code,bm_stock_wt) 
    #single_ret=brinson_attr_stock(stock_weight)
    res = brinson_attr_asset(stock_weight, asset_weight, fund_code, stock_benchmark,  bond_benchmark, freq, version, verbose)
    res.to_csv(os.path.join(wdir, 'brinson_result', f'{fund_code}.csv'), encoding='gbk')
    #brinson_attribution()
    #bm_weight = pd.read_csv(os.path.join(wdir, 'index_weight', '000300.csv'),engine='python', encoding='gbk', index_col=[0])
    #bm_weight_new = load_index_weight(benchmark='000300.SH')
    #pct_chg=get_stocks_ret(stock_weight,freq='6M')
    '''
    ret = pd.read_csv(os.path.join(wdir, 'quote_data', '000012.SH.csv'), parse_dates=True,engine='python', encoding='gbk', index_col=[0])
    ret = ret.dropna(how='any', axis=0)['pct_change']
    ret_new = get_index_ret('000012.SH', freq='6M')
    ret_new_300 = get_index_ret('000300.SH', freq='6M')
    '''
    endtime = time.time()
    dtime = endtime - starttime
    print("程序运行时间：%.8s s" % dtime) 