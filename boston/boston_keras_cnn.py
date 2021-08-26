import tensorflow as tf
import pandas as pd
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler

boston_housing = tf.keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# for visualization
# column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
# df = pd.DataFrame(train_data, columns=column_names)
# df.head()

# normalize
scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

    
def build_conv1D_model(train_data_reshaped):
    n_features = train_data_reshaped.shape[1] #13
    n_dim  = train_data_reshaped.shape[2] #1 
    model = keras.Sequential(name="model_conv1D")
    model.add(keras.layers.Input(shape=(n_features,n_dim)))
    model.add(keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu', name="Conv1D_1"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', name="Conv1D_2"))
    model.add(keras.layers.Conv1D(filters=16, kernel_size=2, activation='relu', name="Conv1D_3"))
    model.add(keras.layers.MaxPooling1D(pool_size=2, name="MaxPooling1D"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32, activation='relu', name="Dense_1"))
    model.add(keras.layers.Dense(n_dim, name="Dense_2"))
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',optimizer=optimizer,metrics=['mae'])
    return model

# MLP
def build_model(train_data):
    model = keras.Sequential([
        keras.layers.Input(shape=(train_data.shape[1],)),                  
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ], name="MLP_model")
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
    return model

def test01():
    model = build_model(train_data)
    EPOCHS = 500
    # Store training stats
    history = model.fit(train_data, train_labels, epochs=EPOCHS, validation_split=0.2, verbose=0)
    model.evaluate(test_data, test_labels, verbose=1)
    # 2.82

def test02():

    # reshape data for CNN
    # train_data.shape
    # (404, 13)
    sample_size, num_features = train_data.shape
    input_dimension = 1
    train_data_reshaped = train_data.reshape(sample_size,num_features,input_dimension)
    test_data_reshaped = test_data.reshape(-1,num_features,input_dimension)
    # add one more dimension for CNN
    # (404, 13, 1)

    EPOCHS = 500
    model_conv1D = build_conv1D_model(train_data_reshaped)
    history = model_conv1D.fit(train_data_reshaped, train_labels, epochs=EPOCHS, validation_split=0.2, verbose=0)

    # loss, mae
    model_conv1D.evaluate(test_data_reshaped, test_labels, verbose=1)
    # 2.66

if __name__ == '__main__':
    test01()
    test02()