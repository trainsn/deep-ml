import numpy as np

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for Softmax regression, optimizing parameters with Cross Entropy loss.
	"""
	# Your code here
	n_samples = X.shape[0]
	X_bias = np.hstack([np.ones((n_samples, 1)), X])
	n_features = X_bias.shape[1]

	C = int(np.max(y)) + 1
	W = np.zeros((C, n_features))

	Y_onehot = np.zeros((n_samples, C))
	Y_onehot[np.arange(n_samples), y] = 1

	losses = []
	for i in range(iterations):
		scores = X_bias @ W.T
		scores -= np.max(scores, axis=1, keepdims=True)
		exp_scores = np.exp(scores)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

		loss = -np.sum(Y_onehot * np.log(probs + 1e-15))
		losses.append(float(loss))

		grad = (probs - Y_onehot).T @ X_bias
		W -= learning_rate * grad 
	
	coefficients = W.tolist()
	return coefficients, losses
