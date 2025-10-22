import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
plt.style.use('seaborn-v0_8')

np.random.seed(42)
cepas = ['WT', 'C1', 'C2', 'C3', 'C4', 'C5']
tempos = np.arange(2, 72, 2)  # de 0h até 72h, de 2 em 2h

# Condições de fermentação
ph_values = [6.5, 7.0, 7.5]
temp_values = [30, 35, 40]  # em °C
oxigenacao = [20, 40, 60]   # %

# Gerar as linhas
linhas = []
for cepa in cepas:
    for ph in ph_values:
        for temp in temp_values:
            for o2 in oxigenacao:
                glicose = 50
                phb = 0
                od = 0.05
                for t in tempos:
                    # Simulação com influência de pH, temp e oxigenação
                    consumo = np.random.uniform(0.2, 0.5) * (1 + (temp-35)/100) * (1 + (ph-7)/10)
                    glicose = max(0, glicose - consumo)
                    producao = np.random.uniform(0.05, 0.25) * (1 + (o2-40)/100)
                    phb += producao
                    od += np.random.uniform(0.02, 0.05)

                    linhas.append([cepa, t, ph, temp, o2, glicose, phb, od])

df = pd.DataFrame(linhas, columns=['Cepa', 'Tempo_h', 'pH', 'Temperatura_C', 'Oxigenacao_%', 'Glicose_gL', 'PHB_gL', 'OD600'])

display(df.head(100))
print("\nInformações gerais:")
print(df.info())
df.isnull().sum()
print("\nEstatísticas descritivas:")
display(df.describe())


def tempo_90(grupo: pd.DataFrame) -> float:
    grupo = grupo.sort_values('Tempo_h')
    phb_final = grupo.loc[grupo['Tempo_h'].idxmax(), 'PHB_gL']
    alvo = 0.9 * phb_final
    atingiu = grupo[grupo['PHB_gL'] >= alvo]
    if len(atingiu) == 0:
        return np.nan
    return float(atingiu['Tempo_h'].iloc[0])


def calcular_metricas(df: pd.DataFrame) -> pd.DataFrame:
    # Ordena por tempo
    df_sorted = df.sort_values('Tempo_h')

    # Agrupa por Cepa + Condições
    resumo = (
        df_sorted.groupby(['Cepa', 'pH', 'Temperatura_C', 'Oxigenacao_%'], group_keys=False)
        .apply(lambda x: pd.Series({
            'PHB_final_gL': x.loc[x['Tempo_h'].idxmax(), 'PHB_gL'],
            'Glicose_inicial_gL': x.loc[x['Tempo_h'].idxmin(), 'Glicose_gL'],
            'Glicose_final_gL': x.loc[x['Tempo_h'].idxmax(), 'Glicose_gL'],
            'OD_final': x.loc[x['Tempo_h'].idxmax(), 'OD600']
        }))
    )

    # Glicose consumida e rendimento
    resumo['Glicose_consumida_gL'] = resumo['Glicose_inicial_gL'] - resumo['Glicose_final_gL']
    resumo['Rendimento_PHB_por_gGlicose'] = resumo['PHB_final_gL'] / resumo['Glicose_consumida_gL']

    # T90 (tempo para atingir 90% do PHB final) por grupo
    t90 = df.groupby(['Cepa', 'pH', 'Temperatura_C', 'Oxigenacao_%'], group_keys=False).apply(tempo_90).rename('T90_h')
    resumo = resumo.join(t90)

    # Produtividade média
    tempo_total = df['Tempo_h'].max() - df['Tempo_h'].min()
    resumo['Produtividade_media_gL_h'] = resumo['PHB_final_gL'] / tempo_total

    # Ordena por produção final
    resumo = resumo.sort_values('PHB_final_gL', ascending=False)
    return resumo


resumo = calcular_metricas(df)
display(resumo.head(10))





# Reset índice para ter as colunas acessíveis
resumo_reset = resumo.reset_index()

# Calcular matriz de correlação
corr_vars = ['PHB_final_gL', 'pH', 'Temperatura_C', 'Oxigenacao_%', 'Glicose_consumida_gL', 'Rendimento_PHB_por_gGlicose', 'T90_h', 'Produtividade_media_gL_h']
corr_matrix = resumo_reset[corr_vars].corr()

# Plotar matriz de correlação com matplotlib
fig, ax = plt.subplots(figsize=(8,6))
cax = ax.matshow(corr_matrix, cmap='coolwarm')
plt.colorbar(cax)

# Ajustar ticks com nomes das colunas
ax.set_xticks(range(len(corr_vars)))
ax.set_yticks(range(len(corr_vars)))
ax.set_xticklabels(corr_vars, rotation=90)
ax.set_yticklabels(corr_vars)

# Mostrar os valores das correlações
for (i, j), val in np.ndenumerate(corr_matrix.values):
    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

plt.title('Matriz de Correlação das Métricas')
plt.tight_layout()
plt.show()

# Regressão múltipla
X = resumo_reset[['pH', 'Temperatura_C', 'Oxigenacao_%']]
y = resumo_reset['PHB_final_gL']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print("Coeficientes da regressão múltipla:")
for var, coef in zip(X.columns, model.coef_):
    print(f"  {var}: {coef:.4f}")
print(f"Intercepto: {model.intercept_:.4f}")




r2 = r2_score(y, y_pred)
print(f"\nR² do modelo: {r2:.3f}")

# Gráfico real vs previsto
plt.figure(figsize=(6,4))
plt.scatter(y, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('PHB final real (g/L)')
plt.ylabel('PHB final previsto (g/L)')
plt.title('Real vs Previsto - Regressão Múltipla')
plt.grid(True)
plt.show()




# Selecionar as variáveis preditoras
X = df[["pH", "Temperatura_C", "Oxigenacao_%", "Glicose_gL", "OD600"]].values

# Variável alvo
y = df["PHB_gL"].values

# Embaralhar os índices
np.random.seed(42)
indices = np.random.permutation(len(X))

# Definir tamanho do conjunto de teste
test_size = int(len(X) * 0.2)

# Separar os conjuntos
X_train = X[indices[:-test_size]]
X_test = X[indices[-test_size:]]
y_train = y[indices[:-test_size]]
y_test = y[indices[-test_size:]]


# Resolver por equação normal: beta = (XᵀX)^(-1)Xᵀy
X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]  # Adiciona bias (coluna de 1s)
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
y_pred = X_test_b.dot(theta)


plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("PHB final real (g/L)")
plt.ylabel("PHB final previsto (g/L)")
plt.title("Real vs Previsto - Regressão Linear")
plt.grid(True)
plt.show()



# Calcular a produtividade
df["Produtividade"] = df["PHB_gL"] / df["Tempo_h"]

# Ordenar do mais produtivo para o menos
melhores_condicoes = df.sort_values(by="Produtividade", ascending=False)

# Mostrar as top 10 condições com alta produtividade
melhores_condicoes_top10 = melhores_condicoes[["Cepa", "PHB_gL", "Tempo_h", "Produtividade", "pH", "Temperatura_C", "Oxigenacao_%", "Glicose_gL", "OD600"]].head(10)
display(melhores_condicoes_top10)



theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
# Variáveis usadas no modelo
variaveis = ["Intercepto", "pH", "Temperatura_C", "Oxigenacao_%", "Glicose_gL", "OD600"]

# Exibir os coeficientes
for nome, valor in zip(variaveis, theta):
    print(f"{nome}: {valor:.4f}")




# Lista dos nomes das variáveis e seus coeficientes (theta)
variaveis = ["Intercepto", "pH", "Temperatura_C", "Oxigenacao_%", "Glicose_gL", "OD600"]

# Plotar os coeficientes (exceto o intercepto)
plt.figure(figsize=(8,5))
plt.bar(variaveis[1:], theta[1:], color='cornflowerblue')  # pula o intercepto

plt.title('Coeficientes da Regressão Linear')
plt.ylabel('Peso (influência na PHB_gL)')
plt.xticks(rotation=45)
plt.axhline(0, color='gray', linestyle='--')  # linha horizontal no zero
plt.tight_layout()
plt.show()



# Lista das variáveis sem intercepto
variaveis = ["pH", "Temperatura_C", "Oxigenacao_%", "Glicose_gL", "OD600"]

# Construir a equação como string
equacao = f"PHB_gL = {theta[0]:.4f}"  # Intercepto
for nome, coef in zip(variaveis, theta[1:]):
    sinal = "+" if coef >= 0 else "-"
    equacao += f" {sinal} {abs(coef):.4f} * {nome}"

# Exibir a equação
print("Equação da regressão linear:\n")
print(equacao)


# Seleciona as 10 melhores combinações com maior produtividade
top10 = df.sort_values(by="Produtividade", ascending=False).head(10).copy()

# Reindexa para facilitar o uso nos gráficos
top10.reset_index(drop=True, inplace=True)



plt.figure(figsize=(10, 5))
plt.bar(range(10), top10["Produtividade"], color="royalblue")
plt.xticks(range(10), top10["Cepa"], rotation=45)
plt.xlabel("Cepa (Top 10 combinações)")
plt.ylabel("Produtividade (g/L·h)")
plt.title("Top 10 Condições com Maior Produtividade")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 5))
plt.scatter(top10["Tempo_h"], top10["PHB_gL"], c="darkgreen", s=80)

for i, cepa in enumerate(top10["Cepa"]):
    plt.text(top10["Tempo_h"][i]+0.1, top10["PHB_gL"][i], str(cepa), fontsize=9)

plt.xlabel("Tempo (h)")
plt.ylabel("PHB final (g/L)")
plt.title("Relação PHB vs Tempo nas 10 Melhores Combinações")
plt.grid(True)
plt.tight_layout()
plt.show()




variaveis = ["pH", "Temperatura_C", "Oxigenacao_%", "Glicose_gL"]
cores = ["tomato", "gold", "skyblue", "orchid"]

plt.figure(figsize=(12, 6))
for i, var in enumerate(variaveis):
    plt.subplot(1, 4, i+1)
    plt.bar(range(10), top10[var], color=cores[i])
    plt.xticks(range(10), top10["Cepa"], rotation=90)
    plt.title(var)
    plt.tight_layout()

plt.suptitle("Condições Experimentais das 10 Combinações Mais Produtivas", y=1.05)
plt.show()


# Criar a métrica eficiência
df['eficiencia'] = df['PHB_gL'] / df['Tempo_h']

# Agrupar e calcular a média da eficiência por cepa
eficiencia_por_cepa = df.groupby('Cepa')['eficiencia'].mean().sort_values(ascending=False)
print(eficiencia_por_cepa)

# Plotar
plt.figure(figsize=(8,5))
plt.bar(eficiencia_por_cepa.index, eficiencia_por_cepa.values, color='skyblue')

plt.title('Eficiência média de produção de PHB por hora por Cepa')
plt.xlabel('Cepa')
plt.ylabel('Eficiência (g PHB / hora)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()





