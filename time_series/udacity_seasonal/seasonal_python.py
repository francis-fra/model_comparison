import numpy as np
import matplotlib.pyplot as plt
# import tensorflow.keras

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))
  
def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)
    
def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def get_data():
    time = np.arange(4 * 365 + 1)

    slope = 0.05
    baseline = 10
    amplitude = 40
    series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

    noise_level = 5
    noise = white_noise(time, noise_level, seed=42)
    series += noise

    return time, series

def split_train_test(time, series, split_time=1000):
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]
    return x_train, time_train, x_valid, time_valid

class naive_predictor:
    def __init__(self):
        self.train_series = None
    def fit(self, X):
        self.train_series = X
        return self
    def predict(self, X, t):
        # t is the time index
        # X is the series up to t-1
        series = np.concatenate((self.train_series, X), axis=0)
        return series[t-1]

class moving_avg_predictor:
    def __init__(self, window_size=30):
        self.train_series = None
        self.window_size = window_size
    def fit(self, X):
        self.train_series = X
        return self
    def predict(self, X, t):
        series = np.concatenate((self.train_series, X), axis=0)
        return series[t-self.window_size:t].mean()

class detrend_moving_avg_predictor:
    def __init__(self, window_size=30):
        self.train_series = None
        self.window_size = window_size
    def fit(self, X):
        self.train_series = X
        return self
    def predict(self, X, t):
        series = np.concatenate((self.train_series, X), axis=0)
        # detrend: get the slope
        diff_series = (series[365:] - series[:-365])
        # padded to get the same length
        diff_series_padded = np.concatenate((np.zeros(365), diff_series), axis=0)
        # esimate
        diff_avg = diff_series_padded[t-self.window_size:t].mean()
        return diff_avg + series[t-365]

class detrend_smooth_predictor:
    def __init__(self, window_size=30, smooth_size=11):
        self.train_series = None
        self.window_size = window_size
        self.smooth_size = smooth_size
    def fit(self, X):
        self.train_series = X
        return self
    def predict(self, X, t):
        series = np.concatenate((self.train_series, X), axis=0)
        # detrend: get the slope
        diff_series = (series[365:] - series[:-365])
        # padded to get the same length
        diff_series_padded = np.concatenate((np.zeros(365), diff_series), axis=0)
        # estimate
        diff_avg = diff_series_padded[t-self.window_size:t].mean()
        # smoothing the past
        past = series[t-365-self.smooth_size: t-365].mean()
        return diff_avg + past

def naive_test(x_train, time_train, x_valid, time_valid, look_back=10, look_forward=1):
    predictor = naive_predictor()
    predictor.fit(x_train)
    # observed time series available up to time t
    pred = [predictor.predict(x_valid[:t], t) for t in time_valid]
    return pred

def movavg_test(x_train, time_train, x_valid, time_valid, look_back=10, look_forward=1):
    predictor = moving_avg_predictor()
    predictor.fit(x_train)
    # observed time series available up to time t
    pred = [predictor.predict(x_valid[:t], t) for t in time_valid]
    return pred

def detrended_movavg_test(x_train, time_train, x_valid, time_valid, look_back=10, look_forward=1):
    predictor = detrend_moving_avg_predictor(50)
    predictor.fit(x_train)
    # observed time series available up to time t
    pred = [predictor.predict(x_valid[:t], t) for t in time_valid]
    return pred

def detrended_smooth_test(x_train, time_train, x_valid, time_valid, look_back=10, look_forward=1):
    predictor = detrend_smooth_predictor(50, 5)
    predictor.fit(x_train)
    # observed time series available up to time t
    pred = [predictor.predict(x_valid[:t], t) for t in time_valid]
    return pred

def get_mae(pred, actual):
    return abs(pred-actual).mean()

# ------------------------------------------------------------
def test01():
    time, series = get_data()
    (x_train, time_train, x_valid, time_valid) = split_train_test(time, series)
    pred = naive_test(x_train, time_train, x_valid, time_valid)
    mae = get_mae(pred, x_valid)
    print(mae)

def test02():
    time, series = get_data()
    (x_train, time_train, x_valid, time_valid) = split_train_test(time, series)
    pred = movavg_test(x_train, time_train, x_valid, time_valid)
    mae = get_mae(pred, x_valid)
    print(mae)

def test03():
    time, series = get_data()
    (x_train, time_train, x_valid, time_valid) = split_train_test(time, series)
    pred = detrended_movavg_test(x_train, time_train, x_valid, time_valid)
    mae = get_mae(pred, x_valid)
    print(mae)

def test04():
    time, series = get_data()
    (x_train, time_train, x_valid, time_valid) = split_train_test(time, series)
    pred = detrended_smooth_test(x_train, time_train, x_valid, time_valid)
    mae = get_mae(pred, x_valid)
    print(mae)

if __name__ == '__main__':
    test01()
    test02()
    test03()
    test04()