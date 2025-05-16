from pickle import TRUE
from typing import List, Dict
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import false
from vnpy_portfoliostrategy import StrategyTemplate, StrategyEngine
from vnpy.trader.object import TickData,OrderData
from statsmodels.formula.api import ols
from vnpy.trader.constant import Status,Direction
import statsmodels.tsa.stattools as ts
import time
class MultiOil_update(StrategyTemplate):

    author = "用Python的交易员"
    leg1_symbol = "sc2208.INE"
    leg2_symbol = "sc2209.INE"
    variables = ["leg1_symbol", "leg2_symbol", "open_position_1", "open_position_2", "waiting_open", "waiting_close", "flag", "flag2", "complusary_closing_time", "close_one", "close_two","open_one","open_two"]

    def __init__(
            self,
            strategy_engine: StrategyEngine,
            strategy_name: str,
            vt_symbols: List[str],
            setting: dict
    ):
        super().__init__(strategy_engine, strategy_name, vt_symbols, setting)
        self.leg1_symbol, self.leg2_symbol = vt_symbols
        self.main_time = -10
        self.second_time = -100
        self.send_open_time = 1e7
        self.send_close_time = 1e7
        self.flag = False #判断交易还是建模
        self.flag2 = True #判断是否进入第二次线性回归
        self.waiting_open = False #等待订单成交
        self.waiting_close = False
        self.tracked = False
        self.running = True
        self.open_time = 1e6
        self.close_time = 1e6
        self.chasing = True
        self.send = False
        self.send_track = False
        self.send_track_open = False
        #储存建模所用数据
        self.datetime_1, self.bid_price_1, self.ask_price_1 = [datetime(2022,1,1,0,0,0,000000)], [0], [0]
        self.history_main_half_first = {'datetime': self.datetime_1, 'main_bid': self.bid_price_1,'main_ask': self.ask_price_1}
        self.datetime_2, self.bid_price_2, self.ask_price_2 = [datetime(2022,1,1,0,0,0,000000)], [0], [0]
        self.history_second_half_first = {'datetime': self.datetime_2, 'second_bid': self.bid_price_2,'second_ask': self.ask_price_2}
        #判断开仓状态
        self.open_position_1 = 0 #多一空二
        self.open_position_2 = 0 #多二空一
        #收盘前半小时不开仓
        self.complusary_open_time_day = 142000
        self.complusary_open_time_night = 255000
        self.complusary_open_time = 1e6
        #强制锁仓时间
        self.complusary_closing_time_day = 145000
        self.complusary_closing_time_night = 262000
        self.complusary_closing_time = 1e6
        #计数器
        self.count = 0
        #订单状态
        #self.close_one,self.close_two,self.open_one,self.open_two = [],[],[],[]
        self.not_active = set(
            [Status.SUBMITTING, Status.NOTTRADED, Status.PARTTRADED, Status.CANCELLED, Status.REJECTED])
        self.active = set([Status.ALLTRADED])
        self.all = []

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        print("策略初始化")

    def on_start(self):
        """
        Callback when strategy is started.
        """
        print("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        print("策略停止")
    
    def linear_regression(self):
        self.df_main_first = pd.DataFrame(self.history_main_half_first)
        self.df_second_first = pd.DataFrame(self.history_second_half_first)
        self.model_one = pd.merge(self.df_main_first, self.df_second_first, on='datetime', how='outer')
        self.model_one.fillna(method='ffill', inplace=True)
        self.model_one = self.model_one.dropna(axis='index', how='any')
        self.model_one['price1'] = self.model_one[['main_ask', 'main_bid']].apply(lambda x: (x['main_ask'] + x['main_bid']) / 2, axis=1)
        self.model_one['price2'] = self.model_one[['second_ask', 'second_bid']].apply(lambda x: (x['second_ask'] + x['second_bid']) / 2, axis=1)
        self.lm = ols('price2~price1', data=self.model_one).fit()

    def trade_1(self,amount):
        self.open_one = {'price':self.main_bid_price,'amount':amount,'status':False,'id':'buy','symbol':self.leg1_symbol}
        self.open_two = {'price':self.second_ask_price,'amount':amount,'status':False,'id':'sell','symbol':self.leg2_symbol}
        self.open1_status = {'status':'not active','direction':'long'}
        self.open2_status = {'status':'not active','direction':'short'}
        self.open_time = self.count
        self.open_resid_1 = self.resid2
        self.open_a1 = self.main_ask_price
        self.open_b2 = self.second_bid_price
        self.open_price1 = self.main_bid_price
        self.open_price2 = self.second_ask_price
        self.waiting_open = True
        self.chasing = True
        self.tracked = True
        self.send_open_time = 1e7
        self.open_position_1 += amount
        self.a1,self.b1,self.a2,self.b2 = self.main_ask_price,self.main_bid_price,self.second_ask_price,self.second_bid_price
        self.buffer = self.main_ask_bid+self.second_ask_bid
        print(['开仓', self.resid2, self.diff,"时间",self.tick_time,"合约1买价",self.main_bid_price,"合约2卖价",self.second_ask_price])
        self.final_profit = 0

    def trade_2(self,amount):
        self.open_one = {'price':self.main_ask_price,'amount':amount,'status':False,'id':'sell','symbol':self.leg1_symbol}
        self.open_two = {'price':self.second_bid_price,'amount':amount,'status':False,'id':'buy','symbol':self.leg2_symbol}
        self.open1_status = {'status':'not active','direction':'short'}
        self.open2_status = {'status':'not active','direction':'long'}
        self.open_time = self.count
        self.open_resid_2 = self.resid2
        self.open_a2 = self.second_ask_price
        self.open_b1 = self.main_bid_price
        self.open_price1 = self.main_ask_price
        self.open_price2 = self.second_bid_price
        self.waiting_open = True
        self.chasing = True
        self.tracked = True
        self.send_open_time = 1e7
        self.open_position_2 += amount
        self.a1,self.b1,self.a2,self.b2 = self.main_ask_price,self.main_bid_price,self.second_ask_price,self.second_bid_price
        self.buffer = self.main_ask_bid+self.second_ask_bid
        print(['开仓', self.resid2, self.diff,"时间",self.tick_time,"合约1卖价",self.main_ask_price,"合约2买价",self.second_bid_price])
        self.final_profit = 0

    def closing_1(self,amount):
        self.close_one = {'price':self.main_ask_price,'amount':amount,'status':False,'id':'sell','symbol':self.leg1_symbol}
        self.close_two = {'price':self.second_bid_price,'amount':amount,'status':False,'id':'buy','symbol':self.leg2_symbol}
        self.close1_status = {'status':'not active','direction':'short'}
        self.close2_status = {'status':'not active','direction':'long'}
        self.a1,self.b1,self.a2,self.b2 = self.main_ask_price,self.main_bid_price,self.second_ask_price,self.second_bid_price
        self.buffer = self.main_ask_bid+self.second_ask_bid
        self.close_price1 = self.main_ask_price
        self.close_price2 = self.second_bid_price
        self.close_time = self.count
        self.waiting_close = True
        self.chasing = True
        self.send_close_time = 1e7
        #self.profit = self.close_price1 - self.open_price1 - self.close_price2 + self.open_price2
        self.profit2 = self.main_bid_price - self.second_ask_price -self.open_a1+self.open_b2
        print(["平仓", self.resid2,self.diff,"时间",self.tick_time,"合约1卖价",self.main_ask_price,"合约2买价",self.second_bid_price])
        print(["利润",self.profit2])

    def closing_2(self,amount):
        self.close_one = {'price':self.main_bid_price,'amount':amount,'status':False,'id':'buy','symbol':self.leg1_symbol}
        self.close_two = {'price':self.second_ask_price,'amount':amount,'status':False,'id':'sell','symbol':self.leg2_symbol}
        self.close1_status = {'status':'not active','direction':'long'}
        self.close2_status = {'status':'not active','direction':'short'}
        self.close_time = self.count
        self.waiting_close = True
        self.a1,self.b1,self.a2,self.b2 = self.main_ask_price,self.main_bid_price,self.second_ask_price,self.second_bid_price
        self.buffer = self.main_ask_bid+self.second_ask_bid
        self.close_price1 = self.main_bid_price
        self.close_price2 = self.second_ask_price
        self.chasing = True
        self.send_close_time = 1e7
        #self.profit = self.close_price2-self.open_price2 - self.close_price1 + self.open_price1
        self.profit2 = self.second_bid_price - self.main_ask_price -self.open_a2+self.open_b1
        print(['平仓', self.resid2, self.diff,"时间",self.tick_time,"合约1买价",self.main_bid_price,"合约2卖价",self.second_ask_price])
        print(["利润",self.profit2])
        
    def track_open(self):
        self.trade_open_order()
        if (self.open1_status['status'] == 'active') and (self.open2_status['status']  == 'not active') and (self.count - self.open_time > 480 or abs(self.open_price1 - self.price1)>0.5):
            self.open_two['id'] = ''
            print(['订单一成交',self.count - self.open_time])
            self.send_track_open = True
        elif (self.open2_status['status'] == 'active') and (self.open1_status['status'] == 'not active') and (self.count - self.open_time  > 480 or abs(self.open_price2 - self.price2)>0.5):
            self.open_one['id'] = ''
            print(['订单二成交',self.count - self.open_time])
            self.send_track_open = True
        elif (self.open1_status['status'] == 'not active') and (self.open2_status['status'] == 'not active') and self.count - self.open_time > 480:
            self.open_one['id'] = ''
            self.open_two['id'] = ''
            print(['订单均未成交'])
            self.send_track_open = True
        elif (self.open1_status['status'] == 'active') and (self.open2_status['status']  == 'active') and self.open_one['symbol'] == self.open_two['symbol']:
            print(['订单成交','final profit',self.final_profit],'\n')
            self.waiting_open = False
            self.all.append(self.final_profit)
        elif (self.open1_status['status'] == 'active') and (self.open2_status['status']  == 'active'):
            print(['开仓订单成交',self.count -self.send_open_time])
            self.waiting_open = False

    def send_open(self):
        self.trade_open_order()
        if (self.open1_status['status'] == 'active') and (self.open2_status['status']  == 'not active'):
            num = 1#self.status_2.volume - self.status_2.traded
            if self.open1_status['direction']=='long':
                #self.open_two = self.sell(self.leg1_symbol,self.main_bid_price,num)
                self.open_two = {'price':self.main_bid_price,'amount':num,'status':False,'id':'sell','symbol':self.leg1_symbol}
                self.open2_status['direction'] = 'short'
                if self.tracked == True:
                    self.open_position_1 -= num
                    self.tracked = False
            else:
                #self.open_two = self.cover(self.leg1_symbol,self.main_ask_price,num)
                self.open_two = {'price':self.main_ask_price,'amount':num,'status':False,'id':'buy','symbol':self.leg1_symbol}
                self.open2_status['direction'] = 'long'
                if self.tracked == True:
                    self.open_position_2 -= num
                    self.tracked = False
            self.send_open_time = self.count
            self.send_track_open = False
        elif (self.open1_status['status'] == 'not active') and (self.open2_status['status']  == 'active'):
            num = 1#self.status_1.volume-self.status_1.traded
            if self.open2_status['direction']=='long':
                #self.open_one = self.sell(self.leg2_symbol,self.second_bid_price,num)
                self.open_one = {'price':self.second_bid_price,'amount':num,'status':False,'id':'sell','symbol':self.leg2_symbol}
                self.open1_status['direction'] = 'short'
                if self.tracked == True:
                    self.open_position_2 -= num
                    self.tracked = False
            else:
                #self.open_one = self.cover(self.leg2_symbol,self.second_ask_price,num)
                self.open_one = {'price':self.second_ask_price,'amount':num,'status':False,'id':'buy','symbol':self.leg2_symbol}
                self.open2_status['direction'] = 'long'
                if self.tracked == True:
                    self.open_position_1 -= num
                    self.tracked = False
            self.send_open_time = self.count
            self.send_track_open = False
        elif (self.open1_status['status'] == 'not active') and (self.open2_status['status'] == 'not active') :
            untraded_1 = 1#self.status_1.volume-self.status_1.traded
            untraded_2 = 1#self.status_2.volume-self.status_2.traded
            if untraded_1 > untraded_2:
                num = untraded_1 - untraded_2
                if self.open2_status['direction']=='long':
                    #self.open_one = self.sell(self.leg2_symbol,self.second_bid_price,num)
                    self.open_one = {'price':self.second_bid_price,'amount':num,'status':False,'id':'sell','symbol':self.leg2_symbol}
                    self.open1_status['direction'] = 'short'
                    if self.tracked == True:
                        self.open_position_2 -= untraded_1
                        self.tracked = False
                else:
                    #self.open_one = self.cover(self.leg2_symbol,self.second_ask_price,num)
                    self.open_one = {'price':self.second_ask_price,'amount':num,'status':False,'id':'buy','symbol':self.leg2_symbol}
                    self.open1_status['direction'] = 'long'
                    if self.tracked == True:
                        self.open_position_1 -= untraded_1
                        self.tracked = False
            if untraded_2 > untraded_1:
                num = untraded_2 - untraded_1
                if self.open1_status['direction']=='long':
                    #self.open_two = self.sell(self.leg1_symbol,0,num)
                    self.open_two = {'price':0,'amount':num,'status':False,'id':'sell','symbol':self.leg1_symbol}
                    self.open2_status['direction'] = 'short'
                    if self.tracked == True:
                        self.open_position_1 -= untraded_2
                        self.tracked = False
                else:
                    #self.open_two = self.cover(self.leg1_symbol,self.main_ask_price,num)
                    self.open_two = {'price':self.main_ask_price,'amount':num,'status':False,'id':'buy','symbol':self.leg1_symbol}
                    self.open2_status['direction'] = 'long'
                    if self.tracked == True:
                        self.open_position_2 -= untraded_2
                        self.tracked = False
            if untraded_1 == untraded_2:
                if self.open1_status['direction']=='long':
                    if self.tracked == True:
                        self.open_position_1 -= untraded_1
                        self.tracked = False
                else:
                    if self.tracked == True:
                        self.open_position_2 -= untraded_2
                        self.tracked = False
            self.send_open_time = self.count
            self.send_track_open = False
            
    def track_close(self):
        self.trade_close_order()
        if (self.close1_status['status'] == 'active') and (self.close2_status['status'] == 'not active') and (self.count - self.close_time  > 480 or abs(self.close_price1 - self.price1)>0.5):
            self.close_two['id'] = ''
            self.send_track = True
            print(['订单一成交',self.count - self.send_close_time])
        elif (self.close2_status['status'] == 'active') and (self.close1_status['status'] == 'not active') and (self.count - self.close_time  > 480 or abs(self.close_price2 - self.price2)>0.5):
            self.close_one['id'] = ''
            self.send_track = True
            print(['订单二成交',self.count - self.send_close_time])
        elif (self.close1_status['status'] == 'not active') and (self.close2_status['status'] == 'not active') and (self.count - self.close_time  > 480):
            self.close_one['id'] = ''
            self.close_two['id'] = ''
            self.send_track = True
            print(['订单均未成交'])
        elif (self.close1_status['status'] == 'active') and (self.close2_status['status'] == 'active'):
            if self.close2_status['direction'] == 'long':
                self.open_position_1 = 0
            else:
                self.open_position_2 = 0            
            print(['平仓订单成交',self.count - self.send_close_time])
            print(['final_profit',self.final_profit],'\n')
            self.waiting_close = False
            self.all.append(self.final_profit)

    def send_close(self):
        self.trade_close_order()
        if (self.close1_status['status'] == 'active') and (self.close2_status['status'] == 'not active'):
            if self.close1_status['direction'] == 'long':
                num = 1#self.status_2.volume-self.status_2.traded
                #self.close_two = self.sell(self.leg2_symbol, self.second_bid_price+0.2, self.status_2.volume-self.status_2.traded)
                self.close_two = {'price':self.second_bid_price+0.2,'amount':num,'status':False,'id':'sell','symbol':self.leg2_symbol}
                self.close2_status['direction'] = 'short'
                self.open_position_2 = 0
            else:
                #self.close_two = self.cover(self.leg2_symbol, self.second_ask_price-0.2, self.status_2.volume-self.status_2.traded)
                num = 1#self.status_2.volume-self.status_2.traded
                self.close_two = {'price':self.second_ask_price-0.2,'amount':num,'status':False,'id':'buy','symbol':self.leg2_symbol}
                self.close2_status['direction'] = 'long'
                self.open_position_1 = 0
            self.send_track = False
            self.send_close_time = self.count
        elif (self.close1_status['status'] == 'not active') and (self.close2_status['status'] == 'active'):
            if self.close2_status['direction'] == 'long':
                #self.close_one = self.sell(self.leg1_symbol, self.main_bid_price+0.2, self.status_1.volume-self.status_1.traded)
                num = 1#self.status_1.volume-self.status_1.traded
                self.close_one = {'price':self.main_bid_price+0.2,'amount':num,'status':False,'id':'sell','symbol':self.leg1_symbol}
                self.close1_status['direction'] = 'short'
                self.open_position_1 = 0
            else:
                #self.close_one = self.cover(self.leg1_symbol, self.main_ask_price-0.2, self.status_1.volume-self.status_1.traded) 
                num = 1#self.status_1.volume-self.status_1.traded
                self.close_one = {'price':self.main_ask_price-0.2,'amount':num,'status':False,'id':'buy','symbol':self.leg1_symbol}
                self.close1_status['direction'] = 'long'
                self.open_position_2 = 0
            self.send_track = False
            self.send_close_time = self.count
        elif (self.close1_status['status'] == 'not active') and (self.close2_status['status'] == 'not active'):
            if self.close2_status['direction'] == 'long':
                #self.close_one = self.sell(self.leg1_symbol, self.main_bid_price+0.2, self.status_1.volume-self.status_1.traded)
                num1 = 1#self.status_1.volume-self.status_1.traded
                self.close_one ={'price':self.main_bid_price+0.2,'amount':num1,'status':False,'id':'sell','symbol':self.leg1_symbol}
                self.close1_status['direction'] = 'short'
                #self.close_two = self.cover(self.leg2_symbol, self.second_ask_price-0.2, self.status_2.volume-self.status_2.traded)
                num2 = 1#self.status_2.volume-self.status_2.traded
                self.close_two = {'price':self.second_ask_price-0.2,'amount':num2,'status':False,'id':'buy','symbol':self.leg2_symbol}
                self.close2_status['direction'] = 'long'
                self.open_position_1 = 0
            else:
                self.trading = True
                #self.close_one = self.sell(self.leg2_symbol, self.second_bid_price+0.2, self.status_2.volume-self.status_2.traded)
                num1 = 1#self.status_2.volume-self.status_2.traded
                self.close_one = {'price':self.second_bid_price+0.2,'amount':num1,'status':False,'id':'sell','symbol':self.leg2_symbol}
                self.close1_status['direction'] = 'short'
                #self.close_two = self.cover(self.leg1_symbol, self.main_ask_price-0.2, self.status_1.volume-self.status_1.traded)
                num2 = 1#self.status_1.volume-self.status_1.traded
                self.close_two = {'price':self.main_ask_price-0.2,'amount':num2,'status':False,'id':'buy','symbol':self.leg1_symbol}
                self.close2_status['direction'] = 'long'
                self.open_position_2 = 0
            self.send_track = False
            self.send_close_time = self.count
        print('发单')
        
    def chase_open(self):
        self.trade_open_order()
        if (self.open1_status['status'] == 'active') and (self.open2_status['status'] == 'not active'):
            self.open_two['id'] = ''
            self.chasing = False
            self.send = True
        elif (self.open2_status['status'] == 'active') and (self.open1_status['status'] == 'not active'):
            self.open_one['id'] = ''
            self.chasing = False
            self.send = True
        elif (self.open1_status['status'] == 'active') and (self.open2_status['status'] == 'active'):
            self.send = False
            self.send_open_time = self.count
            self.chasing = False

    def chase_send_open(self): 
        self.trade_open_order()
        if ( self.open1_status['status'] == 'active') and (self.open2_status['status'] == 'not active'):
            if self.open2_status['direction']=='long':
                #self.open_two = self.buy(self.leg2_symbol,self.b2+self.buffer,1)
                self.open_two = {'price':self.b2+self.buffer,'amount':1,'status':False,'id':'buy','symbol':self.leg2_symbol}
            else:
                #self.open_two = self.short(self.leg2_symbol,self.a2-self.buffer,1)
                self.open_two = {'price':self.a2-self.buffer,'amount':1,'status':False,'id':'sell','symbol':self.leg2_symbol}
            # if self.open_one != [] and self.open_two !=[]:
            self.send = False
            self.send_open_time = self.count
        elif (self.open2_status['status'] == 'active') and (self.open1_status['status'] == 'not active'):
            if self.open1_status['direction']=='long':
                #self.open_one = self.buy(self.leg1_symbol,self.b1+self.buffer,1)
                self.open_one =  {'price':self.b1+self.buffer,'amount':1,'status':False,'id':'buy','symbol':self.leg1_symbol}
            else:
                #self.open_one = self.short(self.leg1_symbol,self.a1-self.buffer,1)
                self.open_one = {'price':self.a1-self.buffer,'amount':1,'status':False,'id':'sell','symbol':self.leg1_symbol}
            # if self.open_one != [] and self.open_two !=[]:
            self.send = False
            self.send_open_time = self.count
    
    def chase_close(self):
        self.trade_close_order()
        if (self.close1_status['status'] == 'active') and (self.close2_status['status'] == 'not active'):
            self.close_two['id'] = ''
            self.chasing = False
            self.send = True
        elif (self.close1_status['status'] == 'not active') and (self.close2_status['status'] == 'active'):
            self.close_one['id'] = ''
            self.chasing = False
            self.send = True
        elif (self.close1_status['status'] == 'active') and (self.close2_status['status'] == 'active'):
            self.send = False
            self.send_close_time = self.count
            self.chasing = False

    def chase_send_close(self):
        self.trade_close_order()
        if (self.close1_status['status'] == 'active') and (self.close2_status['status'] == 'not active'):
            if self.close1_status['direction'] == 'long':
                #self.close_two = self.sell(self.leg2_symbol, self.a2-self.buffer, 1)
                self.close_two = {'price':self.a2-self.buffer,'amount':1,'status':False,'id':'sell','symbol':self.leg2_symbol}
                self.close2_status['direction'] = 'short'
            else:
                #self.close_two = self.cover(self.leg2_symbol, self.b2+self.buffer, 1)
                self.close_two = {'price':self.b2+self.buffer,'amount':1,'status':False,'id':'buy','symbol':self.leg2_symbol}
                self.close2_status['direction'] = 'long'
            self.send = False
            self.send_close_time = self.count
        elif (self.close1_status['status'] == 'not active') and (self.close2_status['status'] == 'active'):
            if self.close2_status['direction'] == 'long':
                #self.close_one = self.sell(self.leg1_symbol, self.a1-self.buffer, 1)
                self.close_one = {'price':self.a1-self.buffer,'amount':1,'status':False,'id':'sell','symbol':self.leg1_symbol}
                self.close1_status['direction'] = 'short'
            else:
                #self.close_one = self.cover(self.leg1_symbol, self.b1+self.buffer, 1) 
                self.close_one = {'price':self.b1+self.buffer,'amount':1,'status':False,'id':'buy','symbol':self.leg1_symbol}
                self.close1_status['direction'] = 'long'
            self.send = False
            self.send_close_time = self.count
        
    def trade_open_order(self):
        if self.open_one['symbol'] == self.leg1_symbol:
            self.bid_deal1,self.ask_deal1 = self.main_bid_price,self.main_ask_price
        else: 
            self.bid_deal1,self.ask_deal1 = self.second_bid_price,self.second_ask_price
        if self.open_two['symbol'] == self.leg1_symbol:
            self.bid_deal2,self.ask_deal2 = self.main_bid_price,self.main_ask_price
        else:
            self.bid_deal2,self.ask_deal2 = self.second_bid_price,self.second_ask_price
        if self.bid_deal1 < self.open_one['price'] and self.open_one['id'] == 'buy' and self.open_one['status'] == False:
            self.open_one['status'] = True
            self.open1_status['status'] = 'active'
            self.open1_status['direction'] = 'long'
            self.final_profit -= self.open_one['price']
        if self.ask_deal1 > self.open_one['price'] and self.open_one['id'] == 'sell' and self.open_one['status'] == False:
            self.open_one['status'] = True
            self.open1_status['status'] = 'active'
            self.open1_status['direction'] = 'short'
            self.final_profit += self.open_one['price']
        if self.bid_deal2 < self.open_two['price'] and self.open_two['id'] == 'buy' and self.open_two['status'] == False:
            self.open_two['status'] = True
            self.open2_status['status'] = 'active'
            self.open2_status['direction'] = 'long'
            self.final_profit -= self.open_two['price']
        if self.ask_deal2 > self.open_two['price'] and self.open_two['id'] == 'sell' and self.open_two['status'] == False:
            self.open_two['status'] = True
            self.open2_status['status'] = 'active'
            self.open2_status['direction'] = 'short'
            self.final_profit += self.open_two['price']

    def trade_close_order(self):
        if self.close_one['symbol'] == self.leg1_symbol:
            self.close_bid_deal1,self.close_ask_deal1 = self.main_bid_price,self.main_ask_price       
        else: 
            self.close_bid_deal1,self.close_ask_deal1 = self.second_bid_price,self.second_ask_price
        if self.close_two['symbol'] == self.leg2_symbol:
            self.close_bid_deal2,self.close_ask_deal2 = self.second_bid_price,self.second_ask_price
        else:
            self.close_bid_deal2,self.close_ask_deal2 = self.main_bid_price,self.main_ask_price
        if self.close_bid_deal1 < self.close_one['price'] and self.close_one['id'] == 'buy' and self.close_one['status'] == False:
            self.close_one['status'] = True
            self.close1_status['status'] = 'active'
            self.close1_status['direction'] = 'long'
            self.final_profit -= self.close_one['price']
        if self.close_ask_deal1 > self.close_one['price'] and self.close_one['id'] == 'sell' and self.close_one['status'] == False:
            self.close_one['status'] = True
            self.close1_status['status'] = 'active'
            self.close1_status['direction'] = 'short'
            self.final_profit += self.close_one['price']
        if self.close_bid_deal2 < self.close_two['price'] and self.close_two['id'] == 'buy' and self.close_two['status'] == False:
            self.close_two['status'] = True
            self.close2_status['status'] = 'active'
            self.close2_status['direction'] = 'long'
            self.final_profit -= self.close_two['price']
        if self.close_ask_deal2 > self.close_two['price'] and self.close_two['id'] == 'sell' and self.close_two['status'] == False:
            self.close_two['status'] = True
            self.close2_status['status'] = 'active'
            self.close2_status['direction'] = 'short'
            self.final_profit += self.close_two['price']

    def on_tick(self, tick: TickData):
        self.tick_time = float(datetime.strftime(tick.datetime, '%H%M%S.%f'))
        if 0 <= self.tick_time <= 23000:
            self.tick_time += 240000
        if 90000 <= self.tick_time <= 151000:
            self.complusary_closing_time = self.complusary_closing_time_day
            self.complusary_open_time = self.complusary_open_time_day
        if 210000 <= self.tick_time <= 270000:
            self.complusary_closing_time = self.complusary_closing_time_night
            self.complusary_open_time = self.complusary_open_time_night
        self.count += 1
            #计算
        if 90000 <= self.tick_time and self.flag == False and self.waiting_open == False and self.waiting_close == False and self.running == True:
            if tick.vt_symbol == self.leg1_symbol:
                self.datetime_1.append(tick.datetime)
                self.bid_price_1.append(tick.bid_price_1)
                self.ask_price_1.append(tick.ask_price_1)
            else:
                self.datetime_2.append(tick.datetime)
                self.bid_price_2.append(tick.bid_price_1)
                self.ask_price_2.append(tick.ask_price_1)
            if len(self.datetime_1)>3600:
                self.linear_regression()
                p = ts.adfuller(self.lm.resid, 1)[1]
                #print([p,np.std(self.lm.resid)])
                if p<0.01:
                    self.flag = True
                    self.mean = np.mean(self.lm.resid)
                    self.std = np.std(self.lm.resid)

        #交易
        if tick.vt_symbol == self.leg1_symbol:
            self.price1=(tick.ask_price_1+tick.bid_price_1)/2
            self.main_ask_bid=tick.ask_price_1-tick.bid_price_1
            self.main_bid_price=tick.bid_price_1
            self.main_ask_price=tick.ask_price_1
            self.main_time=tick.datetime
            self.main_bid_price5 = tick.bid_price_5
            self.main_ask_price5 = tick.ask_price_5
        else:
            self.price2 = (tick.ask_price_1 + tick.bid_price_1) / 2
            self.second_ask_bid=tick.ask_price_1-tick.bid_price_1
            self.second_bid_price=tick.bid_price_1
            self.second_ask_price=tick.ask_price_1
            self.second_time = tick.datetime
            self.second_bid_price5 = tick.bid_price_5
            self.second_ask_price5 = tick.ask_price_5
        if self.main_time == self.second_time and self.running == True:# and self.inited == True and self.trading == True:
            if self.flag == True and self.waiting_open == False and self.waiting_close == False:
                self.resid = self.price2 - self.lm.params['price1'] * self.price1 - self.lm.params['Intercept']
                self.resid2 = (self.resid - self.mean) / self.std
                self.diff = self.price2 - self.price1
                #print([self.resid2,self.diff])
                #强制平仓
                if self.tick_time > self.complusary_closing_time:  
                    if self.open_position_1 > 0:
                        self.closing_1(self.open_position_1)
                    if self.open_position_2 > 0:
                        self.closing_2(self.open_position_2)
                if self.main_ask_bid+self.second_ask_bid < 0.5:
                    if self.resid2 < -(0.6/self.std) and self.open_position_2 == 0 and self.open_position_1 == 0 and self.tick_time <=  self.complusary_open_time:
                        self.trade_2(1)
                    if self.open_position_2 > 0:
                        if self.resid2 - self.open_resid_2 > (0.8/self.std):
                            self.closing_2(1)    
                    if self.resid2 > (0.6/self.std) and self.open_position_1 == 0 and self.open_position_2 == 0 and self.tick_time <=  self.complusary_open_time:
                        self.trade_1(1)
                    if self.open_position_1 > 0:
                        if self.open_resid_1 - self.resid2 > (0.8/self.std):
                            self.closing_1(1)
                    if self.open_position_1 > 0:
                        if 21600 < self.count-self.open_time <= 28800 or (self.resid2 > (3.4/self.std)): 
                            if self.main_bid_price - self.second_ask_price -self.open_a1+self.open_b2 >=0.1:
                                self.closing_1(1)
                                if self.flag2 == True:
                                    self.flag = False
                                    self.flag2 = False
                                    self.datetime_1.clear()
                                    self.bid_price_1.clear()
                                    self.ask_price_1.clear()
                                    self.datetime_2.clear()
                                    self.bid_price_2.clear()
                                    self.ask_price_2.clear()
                                    self.datetime_1.append(datetime(2022,1,1,0,0,0,000000))
                                    self.bid_price_1.append(0)
                                    self.ask_price_1.append(0)
                                    self.datetime_2.append(datetime(2022,1,1,0,0,0,000000))
                                    self.bid_price_2.append(0)
                                    self.ask_price_2.append(0)
                                else:self.running = False
                        if self.count-self.open_time > 28800:
                            self.closing_1(1) 
                            if self.flag2 == True:
                                self.flag = False
                                self.flag2 = False
                                self.datetime_1.clear()
                                self.bid_price_1.clear()
                                self.ask_price_1.clear()
                                self.datetime_2.clear()
                                self.bid_price_2.clear()
                                self.ask_price_2.clear()
                                self.datetime_1.append(datetime(2022,1,1,0,0,0,000000))
                                self.bid_price_1.append(0)
                                self.ask_price_1.append(0)
                                self.datetime_2.append(datetime(2022,1,1,0,0,0,000000))
                                self.bid_price_2.append(0)
                                self.ask_price_2.append(0)
                            else:self.running = False
                    if self.open_position_2 > 0:
                        if 21600 < self.count-self.open_time < 28800:
                            if self.second_bid_price - self.main_ask_price -self.open_a2+self.open_b1 >=0.1:
                                self.closing_2(1)
                                if self.flag2 == True:
                                    self.flag = False
                                    self.flag2 = False
                                    self.datetime_1.clear()
                                    self.bid_price_1.clear()
                                    self.ask_price_1.clear()
                                    self.datetime_2.clear()
                                    self.bid_price_2.clear()
                                    self.ask_price_2.clear()
                                    self.datetime_1.append(datetime(2022,1,1,0,0,0,000000))
                                    self.bid_price_1.append(0)
                                    self.ask_price_1.append(0)
                                    self.datetime_2.append(datetime(2022,1,1,0,0,0,000000))
                                    self.bid_price_2.append(0)
                                    self.ask_price_2.append(0)
                                else:self.running = False
                        if self.count-self.open_time > 28800 or (self.resid2 < (-3.4/self.std)):
                            self.closing_2(1)
                            if self.flag2 == True:
                                self.flag = False
                                self.flag2 = False
                                self.datetime_1.clear()
                                self.bid_price_1.clear()
                                self.ask_price_1.clear()
                                self.datetime_2.clear()
                                self.bid_price_2.clear()
                                self.ask_price_2.clear()
                                self.datetime_1.append(datetime(2022,1,1,0,0,0,000000))
                                self.bid_price_1.append(0)
                                self.ask_price_1.append(0)
                                self.datetime_2.append(datetime(2022,1,1,0,0,0,000000))
                                self.bid_price_2.append(0)
                                self.ask_price_2.append(0)
                            else:self.running = False


            if self.waiting_open == True:
                if self.count - self.open_time > 0 and self.send == True:
                    self.chase_send_open()
                if self.count - self.open_time > 0 and self.chasing == True:
                    self.chase_open() 
                if self.send_track_open == True:
                    self.send_open()
                if self.count - self.open_time  > 0 and self.count - self.send_open_time > 0:
                    self.track_open()
            if self.waiting_close == True:
                if self.count - self.close_time > 0 and self.send == True:
                    self.chase_send_close()
                if self.count - self.close_time > 0 and self.chasing == True:
                    self.chase_close()
                if self.send_track == True:
                    self.send_close()
                if self.count - self.close_time  > 0 and self.count - self.send_close_time > 0:
                    self.track_close()
