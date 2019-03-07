import pickle
import numpy as np
from scipy.special import expit
from histogram import gradient, magnitude_orientation, hog, visualise_histogram

"""
def feature_extraction_hog(x_data):

    N=np.shape(x_data)[0]

    data_hog=np.zeros((N,784))

    for i in range(0,N):
        img_hog=np.reshape(x_data[i],(56,56))
        img_hog=hog(img_hog, cell_size=(2, 2), cells_per_block=(1, 1), visualise=False, nbins=1, signed_orientation=False, normalise=True,flatten=True)
        img_hog=np.reshape(img_hog,(1,784))
        data_hog[i]=img_hog
        print(i)

    return data_hog

"""

def main():


    with open('train.pkl', 'rb') as f:
        data = pickle.load(f)

    x=data[0]
    y=data[1]


    #x=feature_extraction_hog(x)


    D=x.shape[1]

    hidden_size=55
    output_size=36

    w1= np.random.uniform(low=-0.1, high=0.1, size=(hidden_size,(D+1)) )
    w2= np.random.uniform(low=-0.1, high=0.1, size=(output_size, (hidden_size+1)))


    w1, w2= fit(x,y,True,output_size,hidden_size,w1,w2,0.2,0.2)

    np.savetxt('w1.txt', w1)
    np.savetxt('w2.txt', w2)

    y_train_pred = predict(x,w1,w2)

    print(classification_error(y_train_pred, y))




def sigmoid(x):
    return expit(x)


def predict(x, w1, w2):


    bias1=np.ones((x.shape[0],1))
    x_bias=np.concatenate((bias1, x), axis=1)
    output1 = sigmoid(w1 @ np.transpose(x_bias))
    bias2=np.ones((1,output1.shape[1]))
    output1 = np.concatenate((bias2,output1),axis=0)
    output2 = w2 @ output1

    y_pred = np.argmax(output2, axis=0)

    return y_pred

def sigmoid_gradient(z):
        sg = sigmoid(z)
        return sg * (1 - sg)

def add_bias_unit(X, how='column'):

        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        return X_new

def feedforward(X, w1, w2):
        a1 = add_bias_unit(X, how='column')
        z2 = w1.dot(a1.T)
        a2 = sigmoid(z2)
        a2 = add_bias_unit(a2, how='row')
        z3 = w2.dot(a2)
        a3 = sigmoid(z3)
        return a1, z2, a2, z3, a3

def L2_reg(lambda_, w1, w2):
        """Compute L2-regularization cost"""
        return (lambda_ / 2.0) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))

def L1_reg(lambda_, w1, w2):
        """Compute L1-regularization cost"""
        return (lambda_ / 2.0) * (np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sum())


def _encode_labels(y, k):

        onehot = np.zeros((k, y.shape[0]))
        N=y.shape[0]
        for i in range(0,N):
            m=int(y[i,0])
            onehot[m, i] = 1.0

        return onehot

def get_cost(w1, w2, l1, l2, x, y_enc, hidden_size, output_size):


        a1, z2, a2, z3, a3= feedforward(x,w1,w2)

        term1 = -y_enc * (np.log(a3))
        term2 = (1 - y_enc) * np.log(1 - a3)
        cost = np.sum(term1 - term2)
        L1_term = L1_reg(l1, w1, w2)
        L2_term = L2_reg(l2, w1, w2)
        cost = cost + L1_term + L2_term
        return cost

def get_gradient(w1, w2, l1, l2, x, y_enc, hidden_size, output_size):

        a1, z2, a2, z3, a3 = feedforward(x,w1,w2)
        # backpropagation
        sigma3 = a3 - y_enc
        z2 = add_bias_unit(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * sigmoid_gradient(z2)
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)
        grad2 = sigma3.dot(a2.T)

        # regularize
        grad1[:, 1:] += (w1[:, 1:] * (l1 + l2))
        grad2[:, 1:] += (w2[:, 1:] * (l1 + l2))

        return grad1, grad2


def classification_error(predict, y_true):
    """
    Function calculates classification error
    :param p_y_x: matrix of predicted probabilities
    :param y_true: set of ground truth labels 1xN.
    Each row of matrix represents distribution p(y|x)
    :return: classification error
    """

    N1=len(predict)
    sum=0

    for i in range(0,N1):
        if(predict[i]!=y_true[i]):
            sum=sum+1

    sum=sum/N1

    return sum




def fit(X, y, print_progress,output_size, hidden_size, w1, w2,l1,l2):


    eta=0.001
    decrease_constant=0.00001
    epochs = 9000
    shuffle=True
    minibatches=50
    alpha=0.001

    cost_ = []
    X_data, y_data = X.copy(), y.copy()
    y_enc = _encode_labels(y, output_size)

    delta_w1_prev = np.zeros(w1.shape)
    delta_w2_prev = np.zeros(w2.shape)

    for i in range(epochs):

        # adaptive learning rate
        eta /= (1 + decrease_constant * i)

        if print_progress:
            print('\rEpoch: %d/%d' % (i + 1, epochs))


        if shuffle:
            idx = np.random.permutation(y_data.shape[0])
            X_data, y_data = X_data[idx], y_data[idx]

            mini = np.array_split(range(y_data.shape[0]), minibatches)
            for idx in mini:
                # feedforward
                a1, z2, a2, z3, a3 = feedforward(X[idx], w1, w2)
                cost = get_cost(w1, w2, l1, l2, X[idx], y_enc[:, idx], hidden_size, output_size)
                cost_.append(cost)

                # compute gradient via backpropagation
                grad1, grad2 = get_gradient(w1,w2,l1,l2,X[idx],y_enc[:, idx],hidden_size, output_size)

                delta_w1, delta_w2 = eta * grad1, eta * grad2
                w1 -= (delta_w1 + (alpha * delta_w1_prev))
                w2 -= (delta_w2 + (alpha * delta_w2_prev))
                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2
        if(epochs%1000==0):
            np.savetxt('w1.txt', w1)
            np.savetxt('w2.txt', w2)


    return (w1,w2)

if __name__ == '__main__':
    main()
