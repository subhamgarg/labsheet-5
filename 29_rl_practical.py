import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model=Sequential([Dense(24,activation='relu',input_shape=(4,)),Dense(24,activation='relu'),Dense(2)])
model.save("dqn_cartpole.keras"); print("Saved dqn_cartpole.keras")
