# Projeto de Análise de Crédito

Este é um projeto de análise de crédito que utiliza Python 3.10.11 para prever a probabilidade de um cliente se tornar um mau pagador. O projeto utiliza um conjunto de dados que inclui informações sobre os clientes, como sexo, posse de veículo, posse de imóvel, entre outras.

## Pré-requisitos

Certifique-se de ter Python 3.10.11 instalado em seu ambiente antes de executar o código.

```bash
python --version
```

Este projeto também requer a biblioteca Pandas para manipulação de dados e a biblioteca Scikit-Learn para criar um modelo de classificação.

Você pode instalar as dependências usando o seguinte comando:

```bash
pip install pandas scikit-learn
```

Como o projeto foi desenvolvido no Jupyter, será necessário a sua pré-instalação para executá-lo.

## Executando o Projeto

Utilizamos o Jupyter para rodar a aplicação (ia_analisecredito.ipynb), o que facilitará o entendimento no desenvolvimento. Caso queira, poderá rodar em um arquivo .py, porém serão necessários ajustes.


## Dados de Entrada

Os dados de entrada estão dentro do arquivo DadosAnáliseDeCrédito.csv, localizado no diretório baseDb. Este arquivo contém as informações de clientes que tiveram créditos concedidos e a situação do pagamento.

O arquivo possui as seguintes colunas:

- posse_de_veiculo
- posse_de_imovel
- qtd_filhos
- tipo_renda
- educacao
- estado_civil
- tipo_residencia
- idade
- sexo
- tempo_emprego
- possui_celular
- possui_fone_comercial
- possui_fone
- possui_email
- qt_pessoas_residencia
- mau

## Desenvolvimento

### Passo 1: Preparando os Dados

Uma parte fundamental da análise de crédito é preparar os dados para serem usados em um modelo de aprendizado de máquina. Isso inclui a criação de variáveis categóricas, como a codificação one-hot para os dados categóricos, e a divisão do conjunto de dados em dados de treinamento e teste. Vamos preparar nossos dados da seguinte maneira:

```python
x = variáveis_apropriadas.drop('mau', axis=1)
y = variáveis_apropriadas['mau']
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)
```

### Passo 2: Construindo um Modelo de Previsão

Vamos utilizar um modelo de classificação para prever a viabilidade do crédito. Neste exemplo, usaremos o ExtraTreesClassifier da biblioteca Scikit-Learn:

```python
from sklearn.ensemble import ExtraTreesClassifier
modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino)
```
Agora que nosso modelo está treinado, podemos avaliar sua precisão usando os dados de teste:

```python
resultado = modelo.score(x_teste, y_teste)
print("Acurácia:", resultado)
```
Nos teste realizado tive 92% de acertividade no treinamento.

### Passo 3: Realizando previsoes

Agora que nosso modelo está treinado, podemos avaliar sua precisão usando os dados de teste:

```python
# Dados das características
dados_caracteristicas = [[1,                   #qtd_filhos
                          51.72602739726028,   #idade
                          11.271232876712329,  #tempo_emprego
                          1,                   #possui_celular
                          0,                   #possui_fone_comercial
                          0,                   #possui_fone
                          0,                   #possui_email
                          3.0,                 #qt_pessoas_residencia
                          False,               #sexo_M
                          False,               #posse_de_veiculo_Y
                          True,                #posse_de_imovel_Y
                          False,               #tipo_renda_Pensioner
                          False,               #tipo_renda_State servant 
                          False,               #tipo_renda_Student
                          False,               #tipo_renda_Working
                          False,               #educacao_Higher education
                          False,               #educacao_Incomplete higher
                          False,               #educacao_Lower secondary
                          True,                #educacao_Secondary / secondary special
                          True,                #estado_civil_Married
                          False,               #estado_civil_Separated
                          False,               #estado_civil_Single / not married
                          False,               #estado_civil_Widow
                          True,                #tipo_residencia_House / apartment
                          False,               #tipo_residencia_Municipal apartment 
                          False,               #tipo_residencia_Office apartment
                          False,               #tipo_residencia_Rented apartment
                          False]]              #tipo_residencia_With parents

# Colunas das características
colunas_caracteristicas = ['qtd_filhos',
                           'idade',
                           'tempo_emprego', 
                           'possui_celular', 
                           'possui_fone_comercial',
                           'possui_fone',
                           'possui_email', 
                           'qt_pessoas_residencia', 
                           'sexo_M', 
                           'posse_de_veiculo_Y', 
                           'posse_de_imovel_Y', 
                           'tipo_renda_Pensioner', 
                           'tipo_renda_State servant', 
                           'tipo_renda_Student',
                           'tipo_renda_Working', 
                           'educacao_Higher education', 
                           'educacao_Incomplete higher', 
                           'educacao_Lower secondary',
                           'educacao_Secondary / secondary special',
                           'estado_civil_Married', 
                           'estado_civil_Separated',
                           'estado_civil_Single / not married',
                           'estado_civil_Widow', 
                           'tipo_residencia_House / apartment',
                           'tipo_residencia_Municipal apartment', 
                           'tipo_residencia_Office apartment',
                           'tipo_residencia_Rented apartment',
                           'tipo_residencia_With parents']

# Crie o DataFrame com os dados das características
dados_df = pd.DataFrame(dados_caracteristicas, columns=colunas_caracteristicas)

# Faça a previsão com o modelo
previsao = modelo.predict(dados_df)

print(previsao)
```

Na previsão acima, irá demonstrar que o possível cliente poderá ser um mau pagador

## Conclusão

Neste projeto, exploramos um exemplo de análise de crédito utilizando Python. Demonstra-se como carregar, explorar e preparar os dados, construir um modelo de previsão e realizar previsões com base em características dos candidatos. A análise de crédito é uma aplicação importante de aprendizado de máquina no setor financeiro, e as técnicas apresentadas podem ser adaptadas para conjuntos de dados do mundo real. Lembre-se sempre de realizar uma análise completa dos dados e escolher o modelo adequado para obter resultados confiáveis.