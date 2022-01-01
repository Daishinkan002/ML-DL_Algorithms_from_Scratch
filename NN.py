import numpy as np
from tqdm import tqdm



class FeedForward:
    
    def __init__(self):
        self.weights = None
        self.biases = None
        self.learning_rate = None
        self.activations_array = []
        self.a = []
        self.z = []
        self.cost_array = []
        self.input_shape = None
    
    def __str__(self):
        return_string = 'Feedforward Neural Network \n\n----------------------------------------------------------------\n'
        for i in range(len(self.weights)):
            return_string += 'Dense\tInput shape = ' + str(self.weights[i].shape) + '\t Activation = ' + self.activations_array[i] + '\n----------------------------------------------------------------\n'
        
        return return_string

    
    def activation(self, x, name='sigmoid'):
        '''
            Applies sigmoid over output and returns
        '''
        if(name=='sigmoid'):
            x = np.array(x)
            return 1.0/(1.0 + np.exp(-x))
        if(name=='relu'):
            return np.maximum(0, x)
        
        if(name=='softmax'):
            all_exp = np.exp(x)
            return all_exp / np.sum(all_exp)
    
    def activation_gradients(self, x, name='sigmoid'):

        if(name=='sigmoid'):
            var = self.activation(x, name)
            return var*(1-var)
        if(name=='relu'):
            return x*(x>0)

    def calc_a(self, weights, x, bias):
        '''
            Returns x*w + b
        '''
        # print(len(self.weights))
        # print(len(self.weights[0]))
        return (weights.T@x) + bias
    

    
    def add_dense(self, num_neurons, input_shape=None, activation='sigmoid'):
        '''
            Add Dense Layer of size num_neurons
        '''
        if input_shape:
            self.weights = []
            self.biases = []
            self.weights.append(np.array(np.random.randn(input_shape, num_neurons)))
            self.a.append(np.zeros(input_shape))
            self.z.append(np.zeros(input_shape))
        else:
            self.weights.append(np.random.randn(len(self.weights[-1][0]), num_neurons))

        self.a.append(np.zeros(num_neurons))
        self.z.append(np.zeros(num_neurons))
        self.biases.append(np.random.randn(num_neurons))
        self.activations_array.append(activation)
        
    
    def get_layers_value(self):
        return self.a[1:]
    
    def cost_calc(self, y, y_hat, loss='mse'):
        if(loss=='mse'):
            cost = np.sum((y-y_hat)**2)
            if(cost<0):
                print(y, y_hat)
            return cost


    def forward_pass(self, y_actual=None):
        #i = 1,2,3,4,5
        for i in range(len(self.a)-1):
            self.a[i+1] = self.calc_a(self.weights[i], self.z[i], self.biases[i])
            self.z[i+1] = self.activation(self.a[i+1], self.activations_array[i])
        if(y_actual is not None):
            return self.cost_calc(y_actual, self.z[-1])
    
    def calc_final_gradient(self, y, y_hat, loss='mse'):
        if(loss=='mse'):
            return (y_hat-y)
    
    def get_weights(self):
        return self.weights

    def backward_pass(self, y, y_hat, learning_rate=0.01, batch_size = 1000):
        dw = [0 for i in range(len(self.z))] # dw has only 2 layers
        final_difference = self.calc_final_gradient(y, y_hat)
        weights_updation_array = [np.zeros(t.shape) for t in self.weights]
        bias_updation_array = [np.zeros(t.shape) for t in self.biases]
        dw[-1] = np.array([final_difference * self.activation_gradients(self.a[-1], self.activations_array[-1])]).T
        for i in range(len(dw)-1, 0, -1):
            dw_grad = np.array([self.z[i-1]]).T @ dw[i].T
            if(i!=1):
                dw[i-1] = (self.weights[i-1]@dw[i])*np.array([self.activation_gradients(self.a[i-1], self.activations_array[i-1])]).T
            weights_updation_array[i-1] = dw_grad
            bias_updation_array[i-1] = np.squeeze(dw[i])
            
        return weights_updation_array, bias_updation_array
                
    def shuffle(self, x, y):
        temp = np.hstack((x,y))
        np.random.shuffle(temp)
        x,y = temp[:,:-1], temp[:,-1]
        return x,y
    
    def batch_wise_updation(self, X_batch, y_batch, learning_rate):
        updation_weights = [np.zeros(t.shape) for t in self.weights]
        updation_biases = [np.zeros(t.shape) for t in self.biases]
        
        m = X_batch.shape[0]
        cost = 0
        count = -2
        for j in range(m):
            self.z[0] = X_batch[j]
            cost += self.forward_pass(y_batch[j])
            weights_updation_array, bias_updation_array = self.backward_pass(y_batch[j], self.z[-1], learning_rate, m)
            updation_weights = [nw+dnw for nw, dnw in zip(updation_weights, weights_updation_array)]
            updation_biases = [nb+dnb for nb, dnb in zip(updation_biases, bias_updation_array)]
        self.weights = [w-(1.0*(learning_rate/m))*nw for w, nw in zip(self.weights, updation_weights)]
        self.biases  = [b-(1.0*(learning_rate/m))*nb for b, nb in zip(self.biases, updation_biases)]
        return (1.0*cost)


    def fit(self, X_train, y_train, loss='mse', epochs = 2, batch_size = 1, learning_rate = 0.01, validation_data = None):
        '''
        Expects 
        X_train - numpy array of shape(length_of_data, features)
        y_train - numpy array of shape(length_of_data, num_classes)
        '''
        if(X_train.shape[1] != len(self.a[0])):
            raise ValueError("Input dimensions doesn't match with input shape provided")
        self.learning_rate = learning_rate
        n = X_train.shape[0]        
        for i in range(epochs):
            total_cost_of_epoch=0
            count = 0
            #shuffle feature is still left to include on training data
            X_batches = [X_train[k: k+batch_size] for k in range(0,n,batch_size)]
            y_batches = [y_train[k: k+batch_size] for k in range(0,n,batch_size)]
            for X, y  in list(zip(X_batches, y_batches)):
                # print('Running ', count, 'Batch')
                total_cost_of_epoch += self.batch_wise_updation(X, y, learning_rate)
                # print('Average Cost of ' + str(count) + ' Batch = ', avg_cost_per_batch)
                count += 1
            self.cost_array.append(total_cost_of_epoch)
            print('Epoch', i , 'completed with total MSE cost', total_cost_of_epoch)
    
    
    def predict(self, X_test):
        y_hat = []
        for j in tqdm(range(X_test.shape[0])):
            self.z[0] = X_test[j]
            self.forward_pass()
            y_hat.append(self.z[-1])
        return y_hat

    def check_accuracy(self, X_test, y_test, one_hot_labelled = True):
        y_hat = self.predict(X_test)
        if(one_hot_labelled):
            total_length = len(y_hat)
            y_test_int = [np.argmax(y_i) for y_i in y_test]
            y_hat_int = [np.argmax(y_i) for y_i in y_hat]
            return 1.0*sum(int(x == y) for (x, y) in zip(y_test_int, y_hat_int))/total_length
        
