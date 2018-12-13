import sklearn
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Flatten, MaxPooling1D

def create_model(filters=32,kernel_size=3,units=128):
    
    embedding_dim = 300
    vocab_size = len(word_index) + 1
    maxlen = 100
    
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=maxlen,weights=[embedding_matrix],trainable=False))
    model.add(Conv1D(filters, kernel_size, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(units, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, verbose=False, epochs=2, batch_size=50)

# define the grid search parameters
filters = [32,64,128,1024,10000]
kernel_size = [2,3,5]
units = [64,128]

param_grid = dict(filters=filters,kernel_size=kernel_size,units=units)# 
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

#Run grid search
grid_result = grid.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
