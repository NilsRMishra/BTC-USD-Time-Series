from src.imports import *
from src.data_processing.data_processing import *
from src.feature_engg.feature_engg import *
from src.feature_engg.data_tensor import *
from src.model.train import *

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