import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import class_weight

# configuração para uso da GPU no Google Colab
print("Versão do TensorFlow:", tf.__version__)
print("GPU disponível:", tf.config.list_physical_devices('GPU'))

from google.colab import drive
drive.mount('/content/drive')

# caminho dos dados disponibilizados pela LBB
image_dir = '/content/drive/MyDrive/Liga Brasileira de Bioinformática/2025/fase 2/desafio 1/dataset/LBB_Missao1-1_Dataset'

# caminho dos labels classificados manualmente
labels_file = '/content/drive/MyDrive/Liga Brasileira de Bioinformática/2025/fase 2/desafio 1/labels.txt'

# carregar dados e labels
df_labels = pd.read_csv(labels_file, header=None, names=['filename', 'class'])
print(f"Total de entradas no arquivo de labels: {len(df_labels)}")
print(df_labels.head())

unique_classes = df_labels['class'].unique()
class_to_int = {cls: i for i, cls in enumerate(unique_classes)}
int_to_class = {i: cls for cls, i in class_to_int.items()}
num_classes = len(unique_classes)

all_images = []
all_labels = []

# verificação de todas imagens 
all_image_filenames_in_dir = set(os.listdir(image_dir))

for index, row in df_labels.iterrows():
    filename = row['filename']
    if filename in all_image_filenames_in_dir:
        img_path = os.path.join(image_dir, filename)
        try:
            img = Image.open(img_path).convert('L')             # carrega como tons de cinza
            img_array = np.array(img).astype('float32') / 255.0 # converte para numpy array com tipo float32 e normaliza para [0, 1]
            all_images.append(img_array)
            all_labels.append(class_to_int[row['class']])
        except FileNotFoundError:
            print(f"Erro: Imagem {filename} não encontrada em {image_dir}")
        except Exception as e:
            print(f"Erro ao processar imagem {filename}: {e}")

# converte a lista de numpy arrays em um único array
# garante a forma (num_imagens, 32, 32, 1)
all_images = np.array(all_images).reshape(-1, 32, 32, 1)
all_labels = np.array(all_labels)

# verificando os dados crregados 
print(f"Forma de all_images: {all_images.shape}")
print(f"Tipo de dado de all_images: {all_images.dtype}")
print(f"Valores mínimos e máximos de pixel em all_images: {np.min(all_images)}, {np.max(all_images)}")

print(f"Total de imagens rotuladas carregadas: {len(all_images)}")
print(f"Número de classes: {num_classes}")
print(f"Classes e seus mapeamentos: {class_to_int}")

# contagem de imagens por classe (no dataset rotulado) 
print("\n--- Contagem de imagens por classe (no dataset rotulado) ---")
for cls_name, cls_idx in class_to_int.items():
    count = np.sum(all_labels == cls_idx)
    print(f"Classe '{cls_name}': {count} imagens")


# calcular pesos de classe para lidar com desequilíbrio
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(all_labels),
    y=all_labels # usar todos os labels para calcular os pesos
)
class_weights_dict = dict(enumerate(class_weights))
print("\nPesos de classe calculados:", class_weights_dict)


# divisão dos dados (treino, validação, teste)
min_samples_per_class = np.bincount(all_labels).min()
X_train, X_temp, y_train, y_temp = train_test_split(all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

min_samples_val_test = np.bincount(y_temp).min()
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Imagens de treinamento: {len(X_train)}")
print(f"Imagens de validação: {len(X_val)}")
print(f"Imagens de teste: {len(X_test)}")

# construção de uma CNN com técnicas de regularização para prevenir overfitting
# arquitetura: 
# - 3 blocos convolucionais com filtros crescentes (32 -> 64 -> 128)
# - cada bloco: Conv2D -> LeakyReLU -> MaxPooling -> BatchNorm -> Dropout
# - camada final com 256 nodes antes da classificação
# - Adicionamos regularização L2 em todas as camadas treináveis
def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        ### BLOCO CONVOLUCIONAL 1 ###
        # Conv2D: 32 filtros 5x5 - extrai características básicas
        # kernel_regularizer: L2 para reduzir overfitting nos pesos
        # padding='same': mantém dimensões espaciais
        layers.Conv2D(32, (5, 5), kernel_regularizer=regularizers.l2(0.0001), input_shape=input_shape, padding='same'),
        
        # LeakyReLU: ativação que permite gradientes pequenos para x < 0
        # alpha=0.1: inclinação para valores negativos
        layers.LeakyReLU(alpha=0.1),

        # MaxPooling: reduz dimensionalidade espacial por fator 2
        layers.MaxPooling2D((2, 2)),
        
        # BatchNormalization: normaliza entrada da próxima camada
        # optamos normalizar para acelerar o treinamento e melhorar a estabilidade
        layers.BatchNormalization(),

        # Dropout: desativamos 30% dos nodes aleatoriamente
        # optamos pelo dropout para previnir o overfitting durante treinamento
        layers.Dropout(0.3),

        ### BLOCO CONVOLUCIONAL 2 ###
        # Conv2D: 64 filtros 3x3 - características mais "complexas"
        # pois os filtros menores (3x3) são mais eficientes em camadas profundas
        # mantemos o MaxPooling2D, BatchNormalization e Dropout
        layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.0001), padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        ### BLOCO CONVOLUCIONAL 3 ###
        # Conv2D: 128 filtros 3x3
        # aumentamos o número de filtros para capturar padrões mais abstratos (diferenciar 'cube' de 'star')
        layers.Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(0.0001), padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Dropout mais alto (40%) nas camadas mais profundas
        # optamos por aumentar o dropout pois nesta camada o overfitting é mais provável
        layers.Dropout(0.4),

        ### CAMADA DENSA (antes da classificação final) ###
        # Flatten: converte mapas de características 2D em vetor 1D
        layers.Flatten(),
        
        # Dense: 256 nodes -> camada de representação final
        # nesta camada fazemos a combinação das características extraídas para classificação
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.0001)),
        layers.LeakyReLU(alpha=0.1),
        layers.BatchNormalization(),
        
        # Dropout alto (50%) na camada densa
        # pois camadas densas são mais propensas a overfitting
        layers.Dropout(0.5),

        ### CAMADA DE SAÍDA ###
        # Dense final: num_classes neurônios com softmax
        # aqui usamos Softmax para converter logits em probabilidades
        layers.Dense(num_classes, activation='softmax')
    ])
    ### COMPILAÇÂO DA CNN ###
    # usamos Adam otimizer com learning rate baixo (0.0001)
    # learning rate baixo para treinamento mais estável
    # evita oscilações grandes na função de perda
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) # reduzido para 0.0001
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# construção do modelo (input_shape, num_classes)
model = build_cnn((32, 32, 1), num_classes)
model.summary()


# definição de callbacks
# - adição de EarlyStopping (considerando val_loss)
# - adição de checkpoint
# - adição de função para reduzir o learning rate (considerando val_loss)

# EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=200, # reduzir paciência
    restore_best_weights=True,
    verbose=1     # para registrar quando o EarlyStopping foi ativado
)

# salvar checkpoints do modelo 
# é importante caso o treino seja interrompido
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='/content/drive/MyDrive/Liga Brasileira de Bioinformática/2025/fase 2/desafio 1/best_CNN_classifier_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# redução do learning_rate
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,      # reduz lr pela metade
    patience=10,     # espera por 10 epochs sem melhorar val_loss
    min_lr=0.000001, # taxa de aprendizado mínima
    verbose=1
)

callbacks_list = [early_stopping, model_checkpoint, reduce_lr]

print("\n--- Treinando o Modelo CNN com Callbacks ---")
history = model.fit(X_train, y_train, batch_size=64, # batch_size ajustado (64)
                    epochs=500,                      # aumentar epochs, earlyStopping vai parar caso val_loss estabilize
                    validation_data=(X_val, y_val),
                    class_weight=class_weights_dict,
                    callbacks=callbacks_list,
                    verbose=1)

# avaliação final (avalia o modelo treinado)
print("\n--- Avaliação final do modelo treinado ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Acurácia final no conjunto de teste (último modelo treinado): {accuracy:.4f}")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n--- Relatório de Classificação ---")
report_target_names = [int_to_class[i] for i in range(num_classes)]
print(classification_report(y_test, y_pred, target_names=report_target_names, zero_division=0))

print("\n--- Matriz de Confusão ---")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=report_target_names,
            yticklabels=report_target_names)
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()

# plots do histórico de treinamento
print("\n--- Plotando Histórico de Treinamento ---")
history_dict = history.history

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_dict['accuracy'], label='Acurácia de Treino')
plt.plot(history_dict['val_accuracy'], label='Acurácia de Validação')
plt.title('Acurácia de Treino e Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history_dict['loss'], label='Perda de Treino')
plt.plot(history_dict['val_loss'], label='Perda de Validação')
plt.title('Perda de Treino e Validação')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# plot de imagens e classificações preditas
print("\n--- Plotando Exemplos de Classificações Preditas ---")

num_images_to_plot = 20
if num_images_to_plot > len(X_test):
    num_images_to_plot = len(X_test)

random_indices = np.random.choice(len(X_test), num_images_to_plot, replace=False)

n_cols = 5
n_rows = (num_images_to_plot + n_cols - 1) // n_cols

plt.figure(figsize=(15, n_rows * 3))

for i, idx in enumerate(random_indices):
    image = X_test[idx]
    true_label_idx = y_test[idx]
    true_label_name = int_to_class[true_label_idx]

    prediction_probs = model.predict(np.expand_dims(image, axis=0), verbose=0)
    predicted_label_idx = np.argmax(prediction_probs)
    predicted_label_name = int_to_class[predicted_label_idx]
    confidence = np.max(prediction_probs) * 100

    plt.subplot(n_rows, n_cols, i + 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"True: {true_label_name}\nPred: {predicted_label_name}\n({confidence:.1f}%)",
              color='green' if predicted_label_idx == true_label_idx else 'red',
              fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.show()
