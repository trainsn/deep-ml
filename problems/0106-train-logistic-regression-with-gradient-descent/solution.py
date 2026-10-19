import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	# Your code here
	n_samples = X.shape[0]
	X_b = np.hstack([np.ones((n_samples, 1)), X])
	n_features = X_b.shape[1]

	theta = np.zeros(n_features)
	losses = []

	for _ in range(iterations):
		z = X_b @ theta
		y_hat = 1 / (1 + np.exp(-z))

		eps = 1e-15
		y_hat_c = np.clip(y_hat, eps, 1 - eps)
		loss = -np.sum(y * np.log(y_hat_c) + (1-y) * np.log(1-y_hat_c))
		losses.append(round(float(loss), 4))

		graident = X_b.T @ (y_hat_c - y)
		theta = theta - learning_rate * graident
	
	coeffients = [round(float(t), 4) for t in theta]
	return coeffients, losses
