import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# Carregando os dados
data = pd.read_csv('C:/Users/Softex/Desktop/UFFS/IA/Trabalho Final/StressLevelDataset.csv')

# Filtrando os dados para remover a classe '0'
data = data[data['stress_level'].isin([1, 2])]

# Mostrar as primeiras linhas do dataframe
print("Primeiras linhas do dataframe:")
print(data.head())

# Descrever o dataframe para entender os atributos
print("\nDescrição do dataframe:")
print(data.describe(include='all'))

# Verificar os tipos de dados
print("\nTipos de dados:")
print(data.dtypes)

# Contar o número de amostras
num_samples = data.shape[0]
print(f"\nNúmero de amostras: {num_samples}")

# Verificando os valores únicos na coluna de rótulo após filtrar
unique_values = data['stress_level'].unique()
print(f"Valores únicos na coluna 'stress_level' após filtrar: {unique_values}")

# Separando as features e o target
X = data.drop(columns=['stress_level'])
y = data['stress_level']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinando o modelo
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Fazendo previsões
y_pred = clf.predict(X_test)

# Calculando a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do modelo: {accuracy * 100:.2f}%")

# Exibindo a matriz de confusão com rótulos compreensíveis
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Não Estressado", "Estressado"]
)
plt.title("Matriz de Confusão - Nível de Estresse dos Alunos")
plt.show()
