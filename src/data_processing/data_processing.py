from src.imports import *

class Preprocessing:
	def __init__(self, path=""):
		self.path = path

	def __conditional_fill_drop(self, data=None, threshold=0.1):
		"""
			Cleans up NaN values in the DataFrame based on a threshold.
			Parameters:
				data (pd.DataFrame): The input DataFrame.
				threshold (float): Proportion threshold to decide between filling or dropping NaNs.
			Returns:
				pd.DataFrame: Cleaned DataFrame.
    	"""
		total_rows = data.shape[0]
		num_col = data.select_dtypes(include="number")
		print(num_col.columns)
		for col in data.columns:
			na_ratio = data[col].isna().sum() / total_rows
			if na_ratio > threshold:
				if col in num_col:  # Fill with mean (for numeric columns)
					data[col].fillna(data[col].mean(), inplace=True)
				else:  # For non-numeric, fill with mode
					data[col].fillna(data[col].mode().iloc[0], inplace=True)
			else:
				# Drop rows where this column has NaNs
				data = data.dropna(axis=0)
		
		return data


	def __cleanup(self, data=None):
		"""

		"""
		if(data.isna().sum().sum() == 0):
			print("No null values")
		else:
			data = self.__conditional_fill_drop(data=data, threshold=0.1)
			# pass

		return data


	def preprocess(self):
		print("Preprocessing Data")
		data = pd.read_csv(self.path)
		data = self.__cleanup(data=data)
		data.to_csv("processed/processed_data.csv", index=False)
		# return data


if __name__ == "__main__":
	path = r"data/btc_3m.csv"
	pp = Preprocessing(path=path)
	data = pp.preprocess()
