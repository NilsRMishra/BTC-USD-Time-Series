from src.imports import *


class LeakyReluTransformerEncoderLayer(nn.TransformerEncoderLayer):
	"""
	Initialize the original Transformer Encoder Layer with LeakyReLU instead of ReLU.
	"""
	# Replaces the default activation function (typically ReLU) with LeakyReLU.
	def __init__(self, *args, **kwargs):
		super(LeakyReluTransformerEncoderLayer, self).__init__(*args, **kwargs)
		self.activation = nn.LeakyReLU(negative_slope=0.01)


class WeightedMSELoss(torch.nn.Module):
	"""
	Computes a weighted MSE loss with weights based on target deviation from the mean target value.
	"""
	def __init__(self, alpha=10.0):
		super().__init__()
		self.alpha = alpha

	def forward(self, y_pred, y_true):
		error = y_pred - y_true
		weight = 1 + self.alpha * torch.abs(y_true - y_true.mean())
		return torch.mean(weight * (error ** 2))



class TimeSeriesModel(nn.Module):
	def __init__(self, input_dim, model_dim=64, n_heads=4, num_layers=2, dim_feedforward=512):
		super(TimeSeriesModel, self).__init__()
		self.embedding = nn.Linear(input_dim, model_dim)

		encoder_layer = LeakyReluTransformerEncoderLayer(d_model=model_dim, 
														nhead=n_heads,
														dim_feedforward=dim_feedforward,
														dropout=0.2
														)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.fc = nn.Linear(model_dim, 1)

	def forward(self, x):
		# x: (batch_size, seq_len, input_dim)
		x = self.embedding(x)  # shape: (batch, seq, model_dim)
		x = x.permute(1, 0, 2)  # Transformer input (seq_len, batch, model_dim)
		out = self.transformer(x)
		out = out[-1]  # Use last time step
		return self.fc(out)

		

if __name__ == "__main__":
	pass