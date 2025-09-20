import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
import random
from sklearn.manifold import TSNE

# carregar imagens PNG
def load_images(image_dir, image_size=(32, 32)):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    images = []
    filenames = []

    for fname in image_files:
        path = os.path.join(image_dir, fname)
        img = Image.open(path).convert('L').resize(image_size)
        images.append(np.array(img) / 255.0)
        filenames.append(fname)

    images = np.array(images).astype('float32')
    return images, filenames

# multi-scale autoencoder (uma camada para detalhes finos e outro para estruturas globais)
def build_multiscale_autoencoder(input_shape=(32, 32, 1), latent_dim=128):
    input_img = layers.Input(shape=input_shape)

    # caminho 1: detalhes locais
    x1 = layers.Conv2D(32, 3, padding='same', activation='relu')(input_img)
    x1 = layers.MaxPooling2D()(x1)
    x1 = layers.Conv2D(64, 3, padding='same', activation='relu')(x1)
    x1 = layers.MaxPooling2D()(x1)
    x1 = layers.Flatten()(x1)

    # caminho 2: estruturas globais
    x2 = layers.Conv2D(32, 7, padding='same', activation='relu')(input_img)
    x2 = layers.MaxPooling2D()(x2)
    x2 = layers.Conv2D(64, 7, padding='same', activation='relu')(x2)
    x2 = layers.MaxPooling2D()(x2)
    x2 = layers.Flatten()(x2)

    # combinar os dois caminhos
    merged = layers.Concatenate()([x1, x2])
    latent = layers.Dense(latent_dim, name='latent_vector')(merged)

    # decoder
    x = layers.Dense(4 * 4 * 64)(latent)
    x = layers.Reshape((4, 4, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, activation='relu', padding='same')(x)
    output = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

    # modelos finais
    autoencoder = models.Model(input_img, output, name="multiscale_autoencoder")
    encoder = models.Model(input_img, latent, name="multiscale_encoder")

    return autoencoder, encoder

# plot de reconstrução
def plot_reconstructions(model, images, n=10):
    idx = np.random.choice(len(images), n)
    sample = images[idx]
    recon = model.predict(np.expand_dims(sample, -1))

    plt.figure(figsize=(2 * n, 4))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(sample[i], cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(2, n, i + 1 + n)
        plt.imshow(recon[i].squeeze(), cmap='gray')
        plt.title("Reconstruída")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# visualização de agrupamento
def show_sample_predictions(images, filenames, cluster_ids, sample_size=10):
    indices = random.sample(range(len(images)), sample_size)
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, sample_size, i + 1)
        plt.imshow(images[idx].squeeze(), cmap='gray')
        plt.axis('off')
        plt.title(f"{filenames[idx]}\ncluster_{cluster_ids[idx]}")
    plt.tight_layout()
    plt.show()

# plot t-SNE
def plot_tsne(latents, cluster_ids):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    projection = tsne.fit_transform(latents)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(projection[:, 0], projection[:, 1], c=cluster_ids, cmap='tab10', s=10)
    plt.title("t-SNE dos vetores latentes")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.colorbar(scatter, ticks=range(10), label="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# pipeline principal
def main():
    image_dir = 'dataset/'
    n_clusters = 10

    images, filenames = load_images(image_dir)
    images = np.expand_dims(images, -1)

    # divisão em treino e validação
    X_train, X_val = train_test_split(images, test_size=0.2, random_state=42)

    # construir modelo
    autoencoder, encoder = build_multiscale_autoencoder()  # multi-scale autoencoder latent_dim 128
    autoencoder.compile(optimizer='adam', loss='mse')

    # adicionando early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=50,
        batch_size=128,
        callbacks=[early_stop],
        verbose=2
    )

    # plotar curva de perda
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title("Evolução da perda")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # plotar reconstruções
    plot_reconstructions(autoencoder, images)

    os.makedirs("models", exist_ok=True)
    autoencoder.save("models/multiscale_autoencoder.keras")
    encoder.save("models/multiscale_encoder.keras")
    print("Modelos salvos em 'models/'")

    # bloco para usar os modelos salvos
    #from tensorflow.keras.models import load_model
    #autoencoder = load_model("models/multiscale_autoencoder.keras")
    #encoder = load_model("models/multiscale_encoder.keras") 

    # embeddings latentes + clustering
    latent_vectors = encoder.predict(images)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42) # 42 é a resposta para tudo!
    cluster_ids = kmeans.fit_predict(latent_vectors)

    # visualizar com t-SNE
    plot_tsne(latent_vectors, cluster_ids)
    
    # salvar classificações
    with open('classification_results.csv', 'w') as f:
        for fname, cluster_id in zip(filenames, cluster_ids):
            f.write(f"{fname},cluster_{cluster_id}\n")

    print("Classificações salvas em 'classification_results.csv'")

    # visualizar predições
    show_sample_predictions(images, filenames, cluster_ids)

if __name__ == "__main__":
    main()
