from Preprocessing import Preprocess
from Post_processing import Post_processing
from Generator import Generator
from Discriminator import Discriminator
from GAN import GAN
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
import pandas as pd

def train(data, epochs, learning_rate, batch_size, save_model=True, model_name='generator'):
    n_feature = 48
    preprocesss = Preprocess(data)
    dataset_real = preprocesss.preprocessor()
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset_real)
    hidden_sizes = [164, 164, 164]
    generator = Generator((n_feature,), (n_feature,), hidden_sizes)
    discriminator = Discriminator((n_feature,), hidden_sizes)

    # Create the GAN model
    gan = GAN(generator, discriminator, learning_rate)
    history = gan.train(dataset, epochs, batch_size)
    if save_model == True:
        generator.save_model(model_name)
        preprocesss.save_features()
        dump(scaler, open('scalling.pkl', 'wb'))
    
    model_output_samples = generator.generate_samples(1000)
    synthetic_data = scaler.inverse_transform(model_output_samples)
    synthetic_data = pd.DataFrame(synthetic_data, columns=dataset_real.columns)

    return history, dataset_real, synthetic_data

def generate_samples(trained_model, n_samples):
    output_shape = 48
    my_model = tf.keras.models.load_model(trained_model + '.h5')
    noise = np.random.normal(0, 1, size=(n_samples,) + (output_shape,))
    model_output_samples = my_model.predict(noise)
    postp = Post_processing(model_output_samples)
    generated_samples = postp.process()
    return generated_samples