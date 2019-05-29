from keras.models import load_model

model= load_model('CNN_encoder_100epoch.h5')
print(model.summary())