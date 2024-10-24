import numpy as np
import pandas as pd
from typing import List, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import RocCurveDisplay
import talib as ta
import matplotlib.pyplot as plt
from sklearn.model_selection._split import _BaseKFold
from typing import Tuple, Optional, Generator, Any, Union
from sklearn.metrics import log_loss, accuracy_score, RocCurveDisplay, auc



######################### Auxillary Functions #####################################

def get_volume_bars(prices: np.ndarray, vols: np.ndarray,
                    times: np.ndarray, bar_vol: int) -> np.ndarray:
    bars = np.zeros(shape=(len(prices), 6), dtype=object)
    ind = 0
    last_tick = 0
    cur_volume = 0
    for i in range(len(prices)):
        cur_volume += vols[i]
        if cur_volume >= bar_vol:
            bars[ind][0] = pd.Timestamp(times[i - 1])            # time
            bars[ind][1] = prices[last_tick]                     # open
            bars[ind][2] = np.max(prices[last_tick: i + 1])      # high
            bars[ind][3] = np.min(prices[last_tick: i + 1])      # low
            bars[ind][4] = prices[i]                             # close
            bars[ind][5] = np.sum(vols[last_tick: i + 1])        # volume
            cur_volume = 0
            last_tick = i + 1
            ind += 1
    return bars[:ind]

def getTEvents(gRaw: pd.Series, h: float) -> np.ndarray:
    """Input is close series, output is the index when the CUSUM > h or CUSUM < -h"""

    gRaw = gRaw[~gRaw.index.duplicated(keep='first')]
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff() # Calculate the 
    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)

def get_daily_vol(close: pd.Series, span0: int = 20) -> pd.Series:
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:]) # This would result in the index in the left is tmr, in the right is today. Both index are in the closeserries
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1    # daily returns
    df0 = df0.ewm(span=span0).std()
    return df0

def get_returns(bars: np.ndarray) -> np.ndarray:
    """Calculate the returns of time series"""
    close_prices = pd.Series(bars[:, 4], index=bars[:, 0])
    return (close_prices.diff() / close_prices)[1:, ].astype(float)

def add_vertical_barrier(close: pd.Series, tEvents: np.ndarray, numDays: int) -> pd.Series:
    t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=tEvents[:t1.shape[0]])    # Index is tEvents and t1 is 1 day after tEvents that has in dollar bars
    return t1

def apply_tripple_barrier(close: pd.Series, events: pd.DataFrame,
                                   pt_sl: List, molecule: np.ndarray) -> pd.DataFrame:
    '''
    Labeling observations using tripple-barrier method
    
        Parameters:
            close (pd.Series): close prices of bars
            events (pd.DataFrame): dataframe with columns:
                                   - t1: The timestamp of vertical barrier (if np.nan, there will not be
                                         a vertical barrier)
                                   - trgt: The unit width of the horizontal barriers
            pt_sl (list): list of two non-negative float values:
                          - pt_sl[0]: The factor that multiplies trgt to set the width of the upper barrier.
                                      If 0, there will not be an upper barrier.
                          - pt_sl[1]: The factor that multiplies trgt to set the width of the lower barrier.
                                      If 0, there will not be a lower barrier.
            molecule (np.ndarray):  subset of event indices that will be processed by a
                                    single thread (will be used later)
        
        Returns:
            out (pd.DataFrame): dataframe with columns [pt, sl, t1] corresponding to timestamps at which
                                each barrier was touched (if it happened)
    '''
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    if pt_sl[0] > 0:
        pt = pt_sl[0] * events_['trgt']
    else:
        pt = pd.Series(data=[np.nan] * len(events.index), index=events.index)    # NaNs
    if pt_sl[1] > 0:
        sl = -pt_sl[1] * events_['trgt']
    else:
        sl = pd.Series(data=[np.nan] * len(events.index), index=events.index)    # NaNs
    
    for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
        df0 = close[loc: t1]                                       # path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']     # path returns
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()        # earlisest stop loss
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()        # earlisest profit taking
    return out

def get_events_tripple_barrier(
    close: pd.Series, tEvents: np.ndarray, pt_sl: float, trgt: pd.Series, minRet: float,
    numThreads: int = 1, t1: Union[pd.Series, bool] = False, side: pd.Series = None
) -> pd.DataFrame:
    '''
    Getting times of the first barrier touch
    
        Parameters:
            close (pd.Series): close prices of bars
            tEvents (np.ndarray): np.ndarray of timestamps that seed every barrier (they can be generated
                                  by CUSUM filter for example)
            pt_sl (float): non-negative float that sets the width of the two barriers (if 0 then no barrier)
            trgt (pd.Series): s series of targets expressed in terms of absolute returns
            minRet (float): minimum target return required for running a triple barrier search
            numThreads (int): number of threads to use concurrently
            t1 (pd.Series): series with the timestamps of the vertical barriers (pass False
                            to disable vertical barriers)
            side (pd.Series) (optional): metalabels containing sides of bets
        
        Returns:
            events (pd.DataFrame): dataframe with columns:
                                       - t1: timestamp of the first barrier touch
                                       - trgt: target that was used to generate the horizontal barriers
                                       - side (optional): side of bets
    '''
    if len(tEvents) > 0:
        trgt = trgt.loc[trgt.index.intersection(tEvents)] # Data trgt la daily vol co so ngay it hon close serues
    trgt = trgt[trgt > minRet]
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
    if side is None:
        side_, pt_sl_ = pd.Series(np.array([1.] * len(trgt.index)), index=trgt.index), [pt_sl[0], pt_sl[0]]
    else:
        side_, pt_sl_ = side.loc[trgt.index.intersection(side.index)], pt_sl[:2]
    # Events is top index that > MinRet
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
    df0 = apply_tripple_barrier(close, events, pt_sl_, events.index)
#     df0 = mpPandasObj(func=apply_tripple_barrier, pdObj=('molecule', events.index),
#                       numThreads=numThreads, close=close, events=events, pt_sl=[pt_sl, pt_sl])
    events['t1'] = df0.dropna(how='all').min(axis=1)
    if side is None:
        events = events.drop('side', axis=1)
    return events


def get_bins(close: pd.Series, events: pd.DataFrame, t1: Union[pd.Series, bool] = False) -> pd.DataFrame:
    '''
    Generating labels with possibility of knowing the side (metalabeling)
    
        Parameters:
            close (pd.Series): close prices of bars
            events (pd.DataFrame): dataframe returned by 'get_events' with columns:
                                   - index: event starttime
                                   - t1: event endtime
                                   - trgt: event target
                                   - side (optional): position side
            t1 (pd.Series): series with the timestamps of the vertical barriers (pass False
                            to disable vertical barriers)
        
        Returns:
            out (pd.DataFrame): dataframe with columns:
                                       - ret: return realized at the time of the first touched barrier
                                       - bin: if metalabeling ('side' in events), then {0, 1} (take the bet or pass)
                                              if no metalabeling, then {-1, 1} (buy or sell)
    '''
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1 # Tinh return khi cham barrier
    if 'side' in events_:
        out['ret'] *= events_['side']
    out['bin'] = np.sign(out['ret'])
    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0
    else:
        if t1 is not None:
            vertical_first_touch_idx = events_[events_['t1'].isin(t1.values)].index
            out.loc[vertical_first_touch_idx, 'bin'] = 0
    return out


def drop_labels(labels: pd.DataFrame, min_pct: float = 0.05) -> pd.DataFrame:
    while True:
        df0 = labels['bin'].value_counts(normalize=True)
        if df0.min() > min_pct or df0.shape[0] < 3:
            break
        print('dropped label', df0.argmin(), df0.min())
        labels = labels[labels['bin'] != df0.index[df0.argmin()]]
    return labels


def get_upside_bars_ma(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df['close'] < df['ma']) & (df.shift(-1)['close'] > df.shift(-1)['ma'])]

def get_downside_bars_ma(df: pd.DataFrame) -> np.ndarray:
    return df[(df['close'] > df['ma']) & (df.shift(-1)['close'] < df.shift(-1)['ma'])]


def print_results(rf: RandomForestClassifier, X_test: np.ndarray,
                  y_test: np.ndarray, y_pred: np.ndarray) -> None:
    print(f'RF accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'RF precision: {precision_score(y_test, y_pred)}')
    print(f'RF recall: {recall_score(y_test, y_pred)}')
    RocCurveDisplay.from_estimator(rf, X_test, y_test)

class PurgedKFold(_BaseKFold):
    '''
    Extend KFold class to work with labels that span intervals.
    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), without training samples in between.
    '''
    
    def __init__(
        self, n_splits: int = 3, t1: Optional[pd.Series] = None, pctEmbargo: float = 0.0
    ) -> None:
        if not isinstance(t1, pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo
        
    def split(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for i, j in test_starts:
            t0 = self.t1.index[i]    # start of test set
            test_indices = indices[i: j]
            maxT1Idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            if maxT1Idx < X.shape[0]:    # right train (with embargo)
                train_indices = np.concatenate((train_indices, indices[maxT1Idx + mbrg:]))
            yield train_indices, test_indices

def plot_cv_results(cv: Union[StratifiedKFold, PurgedKFold], clf: Any, X: pd.DataFrame, y: pd.Series) -> None:
    '''
    Plots ROC curve for each iteration of cross-validation together with the mean curve
    and print cv accuracy.
    Based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval
    '''
    for scoring in ['accuracy', 'precision', 'recall']:
        score = cross_val_score(estimator=clf, X=X, y=y, scoring=scoring, cv=cv, n_jobs=-1)
        print(f'CV mean {scoring}: {np.mean(score)}')
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, (train, test) in enumerate(cv.split(X, y)):
        clf.fit(np.array(X)[train], np.array(y)[train])
        viz = RocCurveDisplay.from_estimator(clf, np.array(X)[test], np.array(y)[test], name="ROC fold {}".format(i),
                                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color="b", label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2, alpha=0.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.")

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic")
    ax.legend(loc="lower right")
    plt.show()

######################### Resample volume bar #####################################

coin = 'BTCUSDT'

data_folder_link = '/Users/khai/Desktop/alphanest/alpha-nest-backtest/data/crypto/binance/kline'
df = pd.read_csv(f'{data_folder_link}/{coin}/FULL.csv',parse_dates=['ts'])[['ts','close','volume']]
df.columns = ['datetime','close','volume']


volume_bars = get_volume_bars(df['close'].values, df['volume'].values, df['datetime'].values, 1000)
volume_bars_df = pd.DataFrame(data=volume_bars[:, 1:], index=volume_bars[:, 0],
                           columns=['open', 'high', 'low', 'close', 'volume'])
volume_bars_df = volume_bars_df[~volume_bars_df.index.duplicated(keep='first')]

######################### CUSUM Filtering #####################################
mean_std = get_daily_vol(volume_bars_df['close']).mean()
volume_ret = get_returns(volume_bars)

tEvents = getTEvents(volume_ret, h=mean_std) # CUMSUM FIlter timestep,
chosen_bars_df = volume_bars_df.loc[tEvents, :]
######################### TRIPPLE BARRIER #####################################

t1 = add_vertical_barrier(volume_bars_df['close'], tEvents, numDays=1) # Vertical Barriers

events = get_events_tripple_barrier(close=volume_bars_df['close'], tEvents=tEvents, pt_sl=[1, 1],
                                    trgt=get_daily_vol(volume_bars_df['close']), minRet=0.000,
                                    numThreads=1, t1=t1) # The column t1 of output is record as time hit the pt or sl, NOI CHUNG CHI RECORD EVENT 


######################### LABELING #####################################
labels = get_bins(close=volume_bars_df['close'], events=events, t1=t1) # ham get bins co phan vertical 
print(labels['bin'].value_counts(normalize=True))

labels = drop_labels(labels, min_pct=0.05) # Drop too few labels
print(f'len of tEvents {len(tEvents)}, t1 {len(t1)}, events {len(events)}, labels {len(labels)}')

######################### MA Trading Strategy #####################################
volume_bars_df_ma = volume_bars_df.copy(deep=True)
volume_bars_df_ma['ma'] = volume_bars_df_ma['close'].rolling(30, min_periods=1).mean()


up_timestamps, down_timestamps = get_upside_bars_ma(volume_bars_df_ma), get_downside_bars_ma(volume_bars_df_ma)


side_index = up_timestamps.index.union(down_timestamps.index) # Uninon ca 2 index
side_data = []
for idx in side_index:
    if idx in up_timestamps.index:
        side_data.append(1)
    else:
        side_data.append(-1)
side = pd.Series(data=side_data, index=side_index)
# tEvents = []
events_ma = get_events_tripple_barrier(close=volume_bars_df['close'], tEvents=tEvents, pt_sl=[1, 2],
                                       trgt=get_daily_vol(volume_bars_df['close']), minRet=0.000,
                                       numThreads=1, t1=t1, side=side) # if side is not None mean meta-labeling
events_ma = events_ma.dropna() # events_ma just contain index of events above, the side is the signal of MA Strategy
bins_ma = get_bins(close=volume_bars_df['close'], events=events_ma, t1=t1) # meta-labeling of signal in bin column, ret is the true return when trading the MA

######################### feature engineering #####################################

sine, leadsine = ta.HT_SINE(volume_bars_df_ma.close)

features_df = pd.DataFrame(index=bins_ma.index)
features_df['std'] = volume_bars_df_ma.close.pct_change().rolling(10).std().loc[bins_ma.index]
features_df['dominant_cycle'] = ta.HT_DCPERIOD(volume_bars_df_ma.close).loc[bins_ma.index]
features_df['dominant_phase'] = ta.HT_DCPHASE(volume_bars_df_ma.close).loc[bins_ma.index]
features_df['trend_mode'] = ta.HT_TRENDMODE(volume_bars_df_ma.close).loc[bins_ma.index]
features_df['sine'] = sine.loc[bins_ma.index]
features_df['leadsine'] = leadsine.loc[bins_ma.index]
features_df['sine_gt_leadsine'] = (sine>leadsine).loc[bins_ma.index]
features_df['bin'] = bins_ma['bin']

feature_columns = ['std','dominant_cycle','dominant_phase','trend_mode','sine_gt_leadsine','sine','leadsine']

######################### Performing Cross Validation #####################################
# X = features_df[feature_columns].copy()
# y = features_df['bin'].astype(int)


# cv = PurgedKFold(n_splits=5, t1=t1[X.index], pctEmbargo=0.05)
# clf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', n_jobs=-1, random_state=42)
# plot_cv_results(cv, clf, X, y)



######################### Model Fitting #####################################


# Set the threshold date for splitting
split_date = '2024-01-01'

# Create the train and test splits based on the index
X_train = features_df.loc[features_df.index < split_date, feature_columns].values
y_train = features_df.loc[features_df.index < split_date, 'bin'].values.astype(int)

X_test = features_df.loc[features_df.index >= split_date, feature_columns].values
y_test = features_df.loc[features_df.index >= split_date, 'bin'].values.astype(int)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print_results(rf, X_test, y_test, y_pred)

# Create a DataFrame for better visualization
feature_importances = rf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()  # To show the highest importance at the top
plt.show()

######################### Returns #####################################

bins_ma_test = bins_ma.loc[bins_ma.index >= split_date].copy()
bins_ma_test['y_pred'] = y_pred
bins_ma_test['y_pred'].replace(0, -1, inplace=True)
bins_ma_test['meta_signal_ret'] = bins_ma_test['ret']*bins_ma_test['y_pred']
bins_ma_test['original_MA_cum_ret'] = (1+bins_ma_test['ret']).cumprod()
bins_ma_test['meta_MA_cum_ret'] = (1+bins_ma_test['meta_signal_ret']).cumprod()
bins_ma_test[['original_MA_cum_ret','meta_MA_cum_ret']].plot()
plt.show()
