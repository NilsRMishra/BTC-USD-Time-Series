from src.imports import *
from src.model.model import *
from src.feature_engg.data_tensor import *

class TrainingModel:
	def __init__(self, sequence_length=128, batch_size=512, epochs=10, learning_rate=1e-5, device=""):
		self.sequence_length = sequence_length
		self.batch_size = batch_size
		self.epochs = epochs
		self.learning_rate = learning_rate
		# self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.device=device

	
	def _model_setup(self):
		# criterion = nn.MSELoss()
		criterion = WeightedMSELoss(alpha=5.0)
		model = TimeSeriesModel(input_dim=47, model_dim=128, num_layers=2, dim_feedforward=1024)
		optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
		# optimizer.to(device)
		model.to(self.device)
		return model, criterion, optimizer

	def train_eval_loop(self, data=None, is_train=False, model=None):
		# data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=is_train)
		if(is_train):
			total_loss = 0.0
			model, criterion, optimizer = self._model_setup()
			model.train()
			for epoch in range(self.epochs):
				for seq, labels in data:
					optimizer.zero_grad()
					# print(seq.shape,labels.shape)
					# model.hidden_cell = (torch.zeros(2*2, 95, model.hidden_cell[0].size(2)).to(self.device),
					#                     torch.zeros(2*2, 95, model.hidden_cell[1].size(2)).to(self.device))

					seq = seq.to(self.device)
					labels = labels.squeeze().to(self.device)
					y_pred = model(seq)
					# print(y_pred.size(), labels[:,4].size())
					# break
					loss = criterion(y_pred.squeeze(), labels)
					loss.backward()
					total_loss += loss.item()
					optimizer.step()
					# print(loss)
					# break
				if epoch % 10 == 0:
					print(f'Epoch {epoch} Loss: {total_loss}')
				# break

			return model
		else:
			# model.eval()
			_, criterion, _ = self._model_setup()
			model=model
			model.to(self.device)
			preds = torch.tensor([])
			reals = torch.tensor([])
			reals = reals.to(self.device)
			preds=preds.to(self.device)

			# try_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

			with torch.no_grad():
				test_loss = 0.0
				for seq, labels in data:
					seq = seq.to(self.device)
					labels = labels.squeeze().to(self.device)
					y_pred = model(seq)
					preds = torch.cat((preds, y_pred.squeeze()), dim=0)
					reals = torch.cat((reals, labels))
					test_loss += criterion(y_pred.squeeze(), labels)
					print(test_loss)
					# break

				avg_test_loss = test_loss / len(test_loader)
				print(f'Average Test Loss: {avg_test_loss.item():.4f}')

			preds = preds.to('cpu').numpy()
			reals = reals.to('cpu').numpy()
			len(preds), len(reals)




if __name__ == "__main__":
	path = "processed/processed_data.csv"
	features = ft_en.FeatureEngineering(path=path)
	X_train, X_test, y_train, y_test, features, scaler_y = features.feature_engg()

	c2t = ConvertToTensors(batch_size=1024)
	train_loader = c2t.data_to_tensor(data=(X_train, y_train))
	test_loader = c2t.data_to_tensor(data=(X_test, y_test))
	device = "cuda" if torch.cuda.is_available() else "cpu"
	trainer = TrainingModel(device=device)
	model = trainer.train_eval_loop(data=train_loader, is_train=True)
	trainer.train_eval_loop(data=test_loader, is_train=False, model=model)
	torch.save(model.state_dict(), "saved_model/BTC_TSF.")