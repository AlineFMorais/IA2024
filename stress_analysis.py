import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# Carregando os dados
data = pd.read_csv('C:/Users/Softex/Desktop/UFFS/IA/Trabalho Final/StressLevelDataset.csv')

# Filtrando os dados para remover a classe '0'
data = data[data['stress_level'].isin([1, 2])]

#primeiras linhas do dataframe
print("Primeiras linhas do dataframe:")
print(data.head())

#descrição do dataframe
print("\nDescrição do dataframe:")
print(data.describe(include='all'))

#verificar os tipos de dados
print("\nTipos de dados:")
print(data.dtypes)

#contar o número de amostras
num_samples = data.shape[0]
print(f"\nNúmero de amostras: {num_samples}")

#verificando os valores únicos na coluna de rótulo após filtrar
unique_values = data['stress_level'].unique()
print(f"Valores únicos na coluna 'stress_level' após filtrar: {unique_values}")

#separando as features e o target
X = data.drop(columns=['stress_level'])
y = data['stress_level']

#divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#treinamento do modelo
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

#fazendoas previsões
y_pred = clf.predict(X_test)

#calculando acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do modelo: {accuracy * 100:.2f}%")

#mostrando a matriz de confusão
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Não Estressado", "Estressado"]
)
plt.title("Matriz de Confusão - Nível de Estresse dos Alunos")
plt.show()
