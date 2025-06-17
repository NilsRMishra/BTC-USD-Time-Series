from src.imports import *

class FeatureEngineering:
	def __init__(self, path=""):
		self.path = path

	def __date_based_features(self, data=None):
		print(f"Working on date based features ...")
		data['datetime'] = pd.to_datetime(data['datetime'])  # Convert UNIX or string to datetime
		data.set_index('datetime', inplace=True)
		data['hour'] = data.index.hour
		data['day_of_week'] = data.index.dayofweek
		data['day_of_month'] = data.index.day
		data['month'] = data.index.month
		data['week_of_year'] = data.index.isocalendar().week

		return data

	def __pricing_based_features(self, data=None):
		print(f"Working on pricing based features ...")
		data['price_range'] = data['high'] - data['low']
		data['body_size'] = abs(data['close'] - data['open'])
		data['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
		data['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']

		data['daily_return'] = data['close'].pct_change()
		data['log_return'] = np.log(data['close'] / data['close'].shift(1))

		return data

	def __window_based_features(self, data=None):
		print(f"Working on rolling window features ...")
		window_sizes = [5, 10, 20]

		for w in window_sizes:
			data[f'rolling_mean_{w}'] = data['close'].rolling(window=w).mean()
			data[f'rolling_std_{w}'] = data['close'].rolling(window=w).std()
			data[f'rolling_max_{w}'] = data['close'].rolling(window=w).max()
			data[f'rolling_min_{w}'] = data['close'].rolling(window=w).min()

		# simple and expo moving averages
		data['SMA_10'] = data['close'].rolling(window=10).mean()
		data['EMA_10'] = data['close'].ewm(span=10, adjust=False).mean()

		return data

	def __market_indicators(self, data=None):
		print(f"Working on market indicator features ...")
		data['momentum_10'] = data['close'] - data['close'].shift(10)

		# RSI (Relative Strength Index) - 14-period
		delta = data['close'].diff()
		gain = delta.clip(lower=0)
		loss = -delta.clip(upper=0)
		avg_gain = gain.rolling(window=14).mean()
		avg_loss = loss.rolling(window=14).mean()
		rs = avg_gain / avg_loss
		data['RSI_14'] = 100 - (100 / (1 + rs))

		# volatility
		data['rolling_volatility_14'] = data['log_return'].rolling(window=14).std()
		data['ATR_14'] = (
			pd.concat([
				data['high'] - data['low'],
				abs(data['high'] - data['close'].shift(1)),
				abs(data['low'] - data['close'].shift(1))
			], axis=1).max(axis=1)
		).rolling(window=14).mean()

		return data

	def __volume_based_features(self, data=None):
		print(f"Working on volume based features ...")
		data['volume_change'] = data['volume'].diff()
		data['volatility_volume_ratio'] = data['rolling_volatility_14'] / data['volume'].rolling(window=14).mean()

		data['return_close'] = data['close'].pct_change().fillna(0)
		data['percent_volatility'] = data['return_close'].rolling(window=14).std().fillna(0)

		return data
	
	def __scientific_features(self, data=None):
		# Scientific indicators - to be understood/explored
		print(f"Working on scientific features ...")
		# Bollinger Bands (20-period)
		rolling_mean = data['close'].rolling(window=20).mean()
		rolling_std = data['close'].rolling(window=20).std()
		data['bollinger_upper'] = rolling_mean + (2 * rolling_std)
		data['bollinger_lower'] = rolling_mean - (2 * rolling_std)

		# Candlestick Shape Features
		data['is_bullish'] = data['close'] > data['open']
		data['is_bearish'] = data['close'] < data['open']

		# Lag Features
		lags = [1, 3, 5]
		for lag in lags:
			data[f'close_lag_{lag}'] = data['close'].shift(lag)
			data[f'volume_lag_{lag}'] = data['volume'].shift(lag)

		return data

	def __scale_data(self, train=None, test=None, scaler=None):
		print(f"Working on data scaling ...")
		scaler_X = MinMaxScaler((-1,1))  # need to check for some better generic way
		scaler_y = MinMaxScaler((-1,1))
		X_train, y_train = train
		X_test, y_test = test
		X_train = scaler_X.fit_transform(X_train)  #Scaling the train values
		y_train = scaler_y.fit_transform(y_train.values.reshape(-1,1))
		X_test = scaler_X.transform(X_test)  #Scaling the test values
		y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

		return X_train, X_test, y_train, y_test, scaler_y


	def __create_sequences_fast(self, data=None, seq_len=60, target_index=0):
		print(f"Working on sequence creation ...")
		num_samples = len(data) - seq_len
		# Create 3D array of shape (num_samples, seq_len, num_features)
		X = np.lib.stride_tricks.sliding_window_view(data, (seq_len, data.shape[1]))[:, 0, :, :]
		X = np.delete(X, target_index, axis=2)
		y = data[seq_len:, target_index]
		print(X.shape)
		return X, y

	def __data_splitting(self, data=None, scaler=None, target_index=None):
		print(f"Working on data conversion ...")
		X = data.drop(columns=['close'])
		y = data['close']
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

		X_train, X_test, y_train, y_test, scaler_y = self.__scale_data(train=(X_train, y_train), 
																 test=(X_test, y_test), 
																 scaler=scaler)
		# create sequences of data for TimeSeries Forecasting

		# print(np.concatenate([X_train, y_train], axis=1))

		X_train, y_train = self.__create_sequences_fast(data=np.concatenate([X_train, y_train], axis=1),
                                         seq_len=192, target_index=target_index)
		
		X_test, y_test = self.__create_sequences_fast(data=np.concatenate([X_test, y_test], axis=1),
                                         seq_len=192, target_index=target_index)

		return (X_train, X_test, y_train, y_test), scaler_y

	def feature_engg(self):
		data = pd.read_csv(self.path)
		# Feature Engineering
		data = (data
                  .pipe(self.__date_based_features)
                  .pipe(self.__pricing_based_features)
                  .pipe(self.__window_based_features)
                  .pipe(self.__market_indicators)
				  .pipe(self.__volume_based_features)
				  .pipe(self.__scientific_features))
		data.dropna(inplace=True)
		
		features = list(data.columns)
		target_index = features.index('close')

		# print(features)

		scaler = MinMaxScaler(feature_range=(-1,1))

		# converting to tensor format for PyTorch
		(X_train, X_test, y_train, y_test), scaler_y = self.__data_splitting(data=data, scaler=scaler, target_index=target_index)

		# print(tensor_data.shape)
		return X_train, X_test, y_train, y_test, features, scaler_y




if __name__ == "__main__":
	path = "processed/processed_data.csv"
	features = FeatureEngineering(path=path)
	X_train, X_test, y_train, y_test, features, scaler_y = features.feature_engg()
	print(X_train[:10])