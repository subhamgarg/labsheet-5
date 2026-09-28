import tensorflow as tf
model=tf.keras.models.load_model("dqn_cartpole.keras")
print("Loaded model successfully")
print("Prediction for zero state:",model.predict([[0,0,0,0]],verbose=0))
