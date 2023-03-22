import matplotlib
import matplotlib.pyplot as plt
import ffn
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
matplotlib.use('TkAgg')

def averageByVariable(x):
    result = []
    for idx in range(3, len(x)+1, 3):
        result.append(x[idx - 3 : idx].mean())

    return result

def getLabelArray(prdData, dIdx):
    '''
    Get Label data for evaluate prediction results
    :param prdData: Array of Prediction Results
    :param dIdx: Array of Price, used for calculating Label
    :return: Label Data
    '''
    # make actual value array(Label Value)
    yActDf = pd.DataFrame()
    for rowName, row in prdData.iterrows():
        # designate Range for calculate return
        startDate = rowName
        endDate = rowName + relativedelta(weeks=4)
        if endDate > dIdx.iloc[-1].name:
            break

        dRange = dIdx.loc[startDate: endDate]

        # Calculate Return
        yAct = (dRange.iloc[-1] / dRange.iloc[0] - 1) * 100
        # designate date of label
        yAct.index = [startDate]

        # Append to total Array
        yActDf = pd.concat([yActDf, yAct], axis=0)

    # set label array name as y
    yActDf.columns = ['y']

    return yActDf

def getPct(df, rollingWindow):
    '''
    :param df: dataframe with label data
    :param rollingWindow: PCT average rolling window
    :return: pct dataframe
    '''
    #get Accuracy of each Models
    pct = df.apply(getPctApply, axis=1, result_type='expand')
    #drop label value
    pct = pct.drop(labels=pct.columns[-1], axis=1)
    #calculate 52 weeks rolling average accuracy
    pct_rolling = pct.rolling(rollingWindow, min_periods=1).mean()

    return pct_rolling

def getPctApply(x):
    pct_result = []
    #Ambiguous returns can also have a significant impact on directional accuracy,
    # so handling accuracy of ambiguous returns separately
    for x_each in x:
        if np.abs(x_each) <= 0.5 and np.abs(x.y) <=0.5:
            pct_result.append(1)
        elif (x_each * x.y) >= 0:
            pct_result.append(1)
        else:
            pct_result.append(0)

    return pct_result

def testing_directional(ts, monthSet='M'):
    ts_clip = ts.loc['2013-01-01':]
    #result of backtesting can be differed by point of investing
    #interger monthSet == nth week of month, string monthSet == M:Month End, SM:Middle of the Month
    if type(monthSet)==int:
        ts_clip = ts_clip.groupby(pd.Grouper(freq='M')).nth(monthSet)
    else:
        ts_clip = ts_clip.resample(monthSet).last()
    ts_clip['port_return'] = ts_clip.apply(
        lambda x: np.abs(x.y) if np.sign(x.prd) == np.sign(x.y) else -np.abs(x.y), axis=1)
    ts_clip['port'] = (ts_clip.port_return / 100 + 1).cumprod()

    return ts_clip

def getSlope(x):
    x_seq = np.arange(len(x))
    fit = np.polyfit(x_seq, x, 1)

    return fit[0]

prdData = pd.read_csv('./result.csv', index_col=0, parse_dates=True)
prdData.columns = range(18)
dIdx = pd.read_excel('./data/reinforce_result.xlsx', sheet_name='Kidx', index_col=0, parse_dates=True)

#average each model's prediction value by type of variables
prdData_avg = prdData.apply(averageByVariable, axis=1, result_type='expand')
#prdData_avg = prdData + pd.Timedelta(weeks=1)

#get label array
yActDf = getLabelArray(prdData_avg, dIdx)

#append label to prediction array
data = pd.concat([prdData_avg, yActDf], axis=1)

'''
Options
Method = Average by Variable
Model Scoring Window = 156 Weeks
Model Scoring Method = Pct argmax
'''

#test Volatility Estimation Power
window = 156
#get prediction result
err = getPct(data, window)
#choose BestModel
bestModel = err.apply(lambda x: err.columns[x.argmax()], axis=1).shift(4).dropna()
#bestModel = pd.DataFrame(np.full(len(bestModel), 0), index=bestModel.index)
bestModel = pd.DataFrame(bestModel)
#append prediction value of BestModel
bestModel['prd'] = bestModel.apply(lambda x: data.loc[x.name][x].values[0], axis=1)
#add actual return
bestModel = pd.concat([bestModel, data.y], axis=1)
bestModel['rsd'] = bestModel.apply(lambda x: np.abs(x.prd - x.y), axis=1).shift(4).rolling(26).mean()
bestModel.to_csv('./Final_Estimation.csv')

#get return table
bt_directional = testing_directional(bestModel, 'M')
ax = ffn.PerformanceStats(prices=bt_directional.port)
ax.display()
bt_directional.port.plot()
plt.show()
