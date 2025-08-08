ALMA_V1
Este é um projeto voltado para processamento de texto e aplicação de modelos de 
machine learning. O repositório contém toda a estrutura necessária para 
carregar dados, processar texto, treinar modelos e realizar análises. 🚀

👨‍💻 Principais Funcionalidades
Carregamento de dados: Código para leitura e preparação de dados 
(data_loader.py).
Processamento de texto: Tokenização e manipulação de textos (tokenizer.py e 
02_text_processing.ipynb).
Treinamento de modelo: Scripts para treinar modelos de aprendizado de máquina 
utilizando dados processados (train.py).
Utilitários: Funções auxiliares para manipulação de dados e execução de tarefas
do projeto (utils.py).
Notebooks Jupyter: Exemplos de análise e documentação passo a passo 
(notebooks/).
📂 Estrutura do Projeto
ALMA_V1/
│
├── data/
│   └── the-verdict.txt         # Dados brutos usados no projeto
│
├── notebooks/
│   └── 02_text_processing.ipynb  # Notebook para processamento textual e testes
│
├── src/
│   ├── __init__.py             # Arquivo de inicialização do módulo
│   ├── data_loader.py          # Script para carregamento de dados
│   ├── model.py                # Estrutura para criação de modelos ML
│   ├── tokenizer.py            # Tokenizadores e funções de NLP
│   ├── train.py                # Código para treinamento de modelos
│   └── utils.py                # Funções utilitárias diversas
│
├── tests/
│   └── (diretório para criação de testes unitários)
│
├── .gitignore                  # Ignora arquivos desnecessários no Git
├── README.md                   # Documentação principal do repositório
└── requirements.txt            # Dependências do projeto (instalação rápida)
