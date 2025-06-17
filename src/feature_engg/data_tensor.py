from src.imports import *
from src.feature_engg import feature_engg as ft_en


class StockDataset(Dataset):
	def __init__(self, X, y):
		self.X = torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X
		self.y = torch.tensor(y, dtype=torch.float32) if not torch.is_tensor(y) else y

	def __len__(self):
		# print(len(self.X))
		return len(self.X)-1

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]


class ConvertToTensors:
	def __init__(self, batch_size=0):
		self.batch_size = batch_size

	def data_to_tensor(self, data=None):
		X, y = data
		dataset = StockDataset(X=X, y=y)

		data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

		return data_loader


if __name__ == "__main__":
	path = "processed/processed_data.csv"
	features = ft_en.FeatureEngineering(path=path)
	X_train, X_test, y_train, y_test, features, scaler_y = features.feature_engg()

	c2t = ConvertToTensors(batch_size=1024)
	train_loader = c2t.data_to_tensor(data=(X_train, y_train))
	test_loader = c2t.data_to_tensor(data=(X_test, y_test))

	print(type(train_loader))