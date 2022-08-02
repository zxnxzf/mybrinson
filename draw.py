# -*- coding: utf-8 -*-
"""
from pyecharts.charts import Bar
from pyecharts import options as opts

bar = Bar()
bar.add_xaxis(['毛衣','寸衫',"领带",'裤子',"风衣","高跟鞋","袜子"])
bar.add_yaxis('商家A',[114,55,27,101,125,27,105])
bar.add_yaxis('商家B',[57,134,101,22,69,90,129])
bar.set_global_opts(title_opts=opts.TitleOpts(title="某商场销售情况",subtitle='A和B公司'),
                   toolbox_opts=opts.ToolboxOpts(is_show=True))
bar.set_series_opts(label_opts=opts.LabelOpts(position="top"))
#bar.render_notebook()    # 在 notebook 中展示#生成 html 文件
bar.render() 

"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:15:23 2022

@author: zxnxzf
"""
import os
import pandas as pd
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Bar,Line
from pyecharts.faker import Faker
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType
import math
from pyecharts.commons.utils import JsCode
# 使用 snapshot-selenium 渲染图片
"""
    bar.set_global_opts(title_opts=opts.TitleOpts(title="BHB分解"))
    bar.set_series_opts(
            label_opts=opts.LabelOpts(
                position="center",
                formatter=JsCode(#javascript
                    "function(x){temp=x.data*100;return String(temp.toFixed(1))+'%';}"
                ),
            )
                )
"""
#from snapshot_selenium import snapshot

wdir=os.path.dirname(__file__)
#画BHB单期分解
def drawpye(dat):
    #temp=pd.DataFrame()
    if "总计" in dat.index:
        dat.drop(["总计"],inplace=True)
    #temp.loc["总计"]=dat.loc["总计"]
    #temp.columns=dat.columns
    #dat.drop(10,inplace=True)
    
    bar=Bar()
    x=[]
    for i in range(1,len(dat.index)+1):
        x.append(f"第{i}期")
    y1=list(dat["配置效应(AR)"])
    y2=list(dat["选股效应(SR)"])
    y3=list(dat["交互效应(IR)"])
    bar.add_xaxis(x)
    bar.add_yaxis("配置效应(AR)", y1, stack="stack1")
    bar.add_yaxis("选股效应(SR)", y2, stack="stack1")
    bar.add_yaxis("交互效应(IR)", y3, stack="stack1")
    bar.set_global_opts(title_opts=opts.TitleOpts(title="BHB分解"))
    bar.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    bar.render("./photo/BHB/BHB分解.html")
    
def draw1():
    #主动权重
    filepath=os.path.join(wdir,"brinson_result",f"{fund_code}_res.xlsx")
    dat1=pd.read_excel(filepath,encoding="gbk",sheet_name=[0,1,2])
    #dat2=pd.read_excel(filepath,encoding="gbk",sheet_name="Sheet2")
    
    temp=pd.DataFrame()    
    count=1
    for key,dat in dat1.items():
        temp[f"主动权重{count}"]=dat["组合权重"]-dat["基准权重"]
    #temp["主动权重2"]=dat1["组合权重"]-dat1["基准权重"]
        count+=1
    
    if "总计" in temp.index:
        temp.drop("总计",inplace=True)
    bar=Bar()
    bar.add_xaxis(list(temp.index))
    bar.add_yaxis("第一期", list(temp["主动权重1"]))
    bar.add_yaxis("第二期", list(temp["主动权重2"]))
    bar.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    bar.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),title_opts=opts.TitleOpts(title="主动权重"))
    bar.render("./photo/BHB/主动权重.html")
    
    #基准收益
    temp1=pd.DataFrame()    
    count=1
    for key,dat in dat1.items():
        temp1[f"基准收益{count}"]=dat["基准收益"]
    #temp["主动权重2"]=dat1["组合权重"]-dat1["基准权重"]
        count+=1
    if "总计" in temp1.index:
        temp1.drop("总计",inplace=True)
    bar1=Bar()
    bar1.add_xaxis(list(temp1.index))
    bar1.add_yaxis("第一期", list(temp1["基准收益1"]))
    bar1.add_yaxis("第二期", list(temp1["基准收益2"]))
    bar1.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    bar1.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),title_opts=opts.TitleOpts(title="主动权重"))
    bar1.render("./photo/BHB/基准收益.html")
    
    #配置收益
    temp2=pd.DataFrame()    
    count=1
    for key,dat in dat1.items():
        temp2[f"配置收益{count}"]=dat["配置效应(AR)"]
    #temp["主动权重2"]=dat1["组合权重"]-dat1["基准权重"]
        count+=1
    if "总计" in temp2.index:
        temp2.drop("总计",inplace=True)
    bar2=Bar()
    bar2.add_xaxis(list(temp2.index))
    bar2.add_yaxis("第一期", list(temp2["配置收益1"]))
    bar2.add_yaxis("第二期", list(temp2["配置收益2"]))
    bar2.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    bar2.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),title_opts=opts.TitleOpts(title="主动权重"))
    bar2.render("./photo/BHB/配置收益.html")
    
    #交互收益
    temp3=pd.DataFrame()#主动收益
    temp4=pd.DataFrame()
    count=1
    for key,dat in dat1.items():
        temp3[f"主动收益{count}"]=dat["组合收益"]-dat["基准收益"]
        temp4[f"交互收益{count}"]=dat["交互效应(IR)"]
        count+=1
    if "总计" in temp3.index:
        temp3.drop("总计",inplace=True)
    if "总计" in temp4.index:
        temp4.drop("总计",inplace=True)
        
    #柱状
    bar3=Bar()
    bar3.add_xaxis(list(temp3.index))
    bar3.add_yaxis("交互收益(IR)",list(temp4["交互收益2"]))
    bar3.extend_axis(
        yaxis=opts.AxisOpts(
            axislabel_opts=opts.LabelOpts(formatter="{value}"), interval=0.1
            )
        )
    bar3.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    bar3.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),title_opts=opts.TitleOpts(title="交互收益"))
    
    #线
    line=Line()
    line.add_xaxis(list(temp3.index))
    line.add_yaxis("主动权重",temp["主动权重2"], yaxis_index=1)#去掉yaxis_index会好看一些 但是右边坐标轴没有刻度
    line.add_yaxis("主动收益",temp3["主动收益2"])
    line.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    line.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),title_opts=opts.TitleOpts(title="交互收益"))
    bar3.overlap(line)
    bar3.render("./photo/BHB/交互收益.html")
    
def draw2(fund_code):
    #单期
    
    #基金的实际收益率 和 沪深300的收益率
    #基金权重和收益率 基金基准权重和收益率
    wb=pd.read_excel(os.path.join(wdir,"temp","wb.xlsx"))
    wp=pd.read_excel(os.path.join(wdir,"temp","wp.xlsx"))
    rb=pd.read_excel(os.path.join(wdir,"temp","rb.xlsx"))
    rp=pd.read_excel(os.path.join(wdir,"temp","rp.xlsx"))
    
    count=2#第几期
    ##############基准######################################################################
    #柱状
    bar3=Bar()
    bar3.add_xaxis(list(rb.columns))
    bar3.add_yaxis("收益率",list(rb.iloc[count]))
    bar3.extend_axis(
        yaxis=opts.AxisOpts(
            axislabel_opts=opts.LabelOpts(formatter="{value}"), interval=0.1
            )
        )
    bar3.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    bar3.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),title_opts=opts.TitleOpts(title="基准行业收益率"))
    
    #线
    line=Line()
    line.add_xaxis(list(wb.columns))
    line.add_yaxis("权重",wb.iloc[count], yaxis_index=1)#去掉yaxis_index会好看一些 但是右边坐标轴没有刻度
    line.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    line.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),title_opts=opts.TitleOpts(title="基准行业权重"))
    
    bar3.overlap(line)
    bar3.render("./photo/单期/基准收益率及权重.html")
    
    ##############基金######################################################################
    #柱状
    bar3=Bar()
    bar3.add_xaxis(list(rp.columns))
    bar3.add_yaxis("收益率",list(rp.iloc[count]))
    bar3.extend_axis(
        yaxis=opts.AxisOpts(
            axislabel_opts=opts.LabelOpts(formatter="{value}"), interval=0.1
            )
        )
    bar3.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    bar3.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),title_opts=opts.TitleOpts(title="基金行业收益率"))
    
    #线
    line=Line()
    line.add_xaxis(list(wp.columns))
    line.add_yaxis("权重",wp.iloc[count], yaxis_index=1)#去掉yaxis_index会好看一些 但是右边坐标轴没有刻度
    line.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    line.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),title_opts=opts.TitleOpts(title="基金行业权重"))
    bar3.overlap(line)
    bar3.render("./photo/单期/基金收益率及权重.html")
    
    #################单期分解###############################################################################
    #为啥数字清除不了
    dat=pd.read_csv(os.path.join(wdir,"brinson_result",f"{fund_code}.csv"),encoding="gbk")
    bar2=Bar()
    bar2.add_xaxis(["配置收益","选择收益"])
    bar2.add_yaxis("", [dat.iloc[count]["配置效应(AR)"],dat.iloc[count]["选股效应(SR)"]])
    bar2.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    bar2.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=0)),title_opts=opts.TitleOpts(title="单期Brinson归因.html"))
    bar2.render("./photo/单期/单期Brinson归因.html")

    
    #################主动权重############################################################
    #dat=pd.read_excel(os.path.join(wdir,"brinson_result",f"{fund_code}_res.xlsx"),encoding="gbk",sheet_name=sheet_names[count])
    excel_reader=pd.ExcelFile(os.path.join(wdir,"brinson_result",f"{fund_code}_res.xlsx")) # 指定文件 
    sheet_names = excel_reader.sheet_names # 读取文件的所有表单名，得到列表 
    dat = excel_reader.parse(sheet_name=sheet_names[count]) # 读取表单的内容，i是表单名的索引，
    dat.drop(["总计"],inplace=True)
    bar1=Bar()
    bar1.add_xaxis(list(dat.index))
    bar1.add_yaxis("",list(dat["组合权重"]-dat["基准权重"]))
    bar1.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    bar1.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),title_opts=opts.TitleOpts(title="主动权重"))
    bar1.render("./photo/单期/主动权重.html")
    
    ##########整个excel表格###################################################################
    dat.insert(2,"主动权重",list(dat["组合权重"]-dat["基准权重"]))
    dat.insert(5,"主动收益",list(dat["组合收益"]-dat["基准收益"]))
    dat.to_excel(os.path.join(wdir,"photo","单期",f"{fund_code}.xlsx"),encoding="gbk")
    
    #################各行业Brinson分析##################################################################################
    bar=Bar()
    x=list(dat.index)
    y1=list(dat["配置效应(AR)"])
    y2=list(dat["选股效应(SR)"])
    bar.add_xaxis(x)
    bar.add_yaxis("配置效应(AR)", y1, stack="stack1")
    bar.add_yaxis("选股效应(SR)", y2, stack="stack1")
    bar.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),title_opts=opts.TitleOpts(title="BF分解"))
    bar.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    bar.render("./photo/单期/行业BF分解.html")
    
    #########################多期#############################################################################
    dat=pd.read_csv(os.path.join(wdir,"brinson_result",f"{fund_code}.csv"),encoding="gbk",index_col=[0])
    if not os.path.exists(os.path.join(wdir,"photo","多期")):
        os.mkdir(os.path.join(wdir,"photo","多期"))
    if "总计" in dat.index:
        temp=dat.loc["总计"]
        dat.drop(["总计"],inplace=True)
    bar2=Bar()
    len1=len(list(dat.index))
    x=[]
    for i in range(1,len1+1):#最后一行为总计
        x.append(f"第{i}期")
    bar2.add_xaxis(x)
    bar2.add_yaxis("配置收益", list(dat["配置收益AR\'"]))
    bar2.add_yaxis("选择收益", list(dat["选择收益SR\'"]))
    bar2.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    bar2.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=0)),title_opts=opts.TitleOpts(title="Brinson归因"))
    bar2.render("./photo/多期/多期分Brinson归因.html")
    
    bar2=Bar()
    bar2.add_xaxis(["配置收益","选择收益"])
    bar2.add_yaxis("", [temp["配置收益AR\'"],temp["选择收益SR\'"]])
    bar2.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    bar2.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=0)),title_opts=opts.TitleOpts(title="Brinson归因"))
    bar2.render("./photo/多期/多期总Brinson归因.html")
    
    #######多期在各行业的配置收益和选择收益##############################################################################
    excel_reader=pd.ExcelFile(os.path.join(wdir,"brinson_result",f"{fund_code}_res.xlsx")) # 指定文件 
    sheet_names = excel_reader.sheet_names # 读取文件的所有表单名，得到列表
    dat1=[]
    for i in range(0,3):
        dat = excel_reader.parse(sheet_name=sheet_names[i]) # 读取表单的内容，i是表单名的索引，
        dat.drop(["总计"],inplace=True)
        dat1.append(dat)
    bar=Bar()
    x=list(dat1[0].index)
    y1=[list(dat1[0]["组合权重"]-dat1[0]["基准权重"]),list(dat1[1]["组合权重"]-dat1[1]["基准权重"]),list(dat1[2]["组合权重"]-dat1[2]["基准权重"])]
    y2=[list(dat1[0]["配置效应(AR)"]),list(dat1[1]["配置效应(AR)"]),list(dat1[2]["配置效应(AR)"])]
    y3=[list(dat1[0]["组合收益"]-dat1[0]["基准收益"]),list(dat1[1]["组合收益"]-dat1[1]["基准收益"]),list(dat1[2]["组合收益"]-dat1[2]["基准收益"])]
    y4=[list(dat1[0]["选股效应(SR)"]),list(dat1[1]["选股效应(SR)"]),list(dat1[2]["选股效应(SR)"])]
    bar=Bar()#主动权重
    bar.add_xaxis(x)
    bar.add_yaxis("第一期",y1[0])
    bar.add_yaxis("第二期",y1[1])
    bar.add_yaxis("第三期",y1[2])
    bar.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    bar.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),title_opts=opts.TitleOpts(title="主动权重"))
    bar.render("./photo/多期/多期主动权重.html")
    
    bar1=Bar()
    bar1.add_xaxis(x)
    bar1.add_yaxis("第一期",y2[0])
    bar1.add_yaxis("第二期",y2[1])
    bar1.add_yaxis("第三期",y2[2])
    bar1.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    bar1.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),title_opts=opts.TitleOpts(title="配置效应"))
    bar1.render("./photo/多期/多期配置效应.html")
    
    bar2=Bar()
    bar2.add_xaxis(x)
    bar2.add_yaxis("第一期",y3[0])
    bar2.add_yaxis("第二期",y3[1])
    bar2.add_yaxis("第三期",y3[2])
    bar2.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    bar2.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),title_opts=opts.TitleOpts(title="主动收益"))
    bar2.render("./photo/多期/多期主动收益.html")
    
    bar3=Bar()
    bar3.add_xaxis(x)
    bar3.add_yaxis("第一期",y4[0])
    bar3.add_yaxis("第二期",y4[1])
    bar3.add_yaxis("第三期",y4[2])
    bar3.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    bar3.set_global_opts(xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),title_opts=opts.TitleOpts(title="选股效应"))
    bar3.render("./photo/多期/多期选股效应.html")
    
if __name__=="__main__":#当前模块名为正在被运行的模块
    fund_code="519702.OF"
    stock_benchmark = '000300.SH'
    filepath=os.path.join(wdir,"brinson_result",f"{fund_code}.csv")
    dat=pd.read_csv(filepath,encoding="gbk",index_col=[0])
    #设置第一列为行索引
    version=2
    if version==1:
        drawpye(dat)
        draw1()
    else:
        draw2(fund_code)
    #drawphotodui(dat)
    
