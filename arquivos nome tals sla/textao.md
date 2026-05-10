SISTEMA DE ANÁLISE FORENSE DE IMAGENS — KINGAMBIT

DOCUMENTAÇÃO TÉCNICA

VISÃO GERAL

O Kingambit consiste em um sistema de análise forense digital desenvolvido em linguagem Python. O objetivo primordial da ferramenta é a inspeção de imagens para identificação de vestígios de edição ou geração por meio de inteligência artificial. O sistema integra técnicas tradicionais de processamento de imagem a um modelo de aprendizagem de máquina treinado com um conjunto de 12.500 imagens provenientes de diversos geradores de IA, permitindo uma distinção fundamentada entre conteúdos legítimos e sintéticos. O nome do projeto é uma referência estratégica à eficiência e autoridade no domínio de dados.

FLUXO DE PROCESSAMENTO

O processamento de uma imagem no sistema é segmentado em etapas sequenciais, onde cada fase analisa atributos específicos do arquivo. Os dados resultantes são consolidados em um vetor de características utilizado pelo modelo para a classificação final.

O fluxo estabelecido compreende:
Primeiro, a extração de metadados.
Segundo, a análise de nível de erro (ELA).
Terceiro, a análise forense visual.
Quarto, a busca reversa, que é uma etapa opcional.
Quinto, a classificação por meio do modelo híbrido.

Cada fase desempenha uma função complementar, garantindo a robustez do diagnóstico.

ESTRUTURA DO PROJETO

A arquitetura do sistema é composta pelos seguintes arquivos e módulos:

main.py: Concentra a lógica principal de análise, incluindo a extração de metadados, análise ELA, análise de ruído, Transformada Rápida de Fourier (FFT), análise de gradiente, correlação RGB, detecção de aberração cromática e interface com a SerpApi para busca reversa.

server.py: Implementa o servidor web utilizando o framework Flask. Gerencia o recebimento de arquivos, a comunicação com o módulo de análise e a entrega de resultados ao usuário.

modelozudo.py: Define a arquitetura do modelo de machine learning, baseada na EfficientNet-B0 integrada a características forenses, abrangendo as rotinas de treinamento, validação e teste.

extrair_features.py: Script destinado ao processamento do conjunto de dados, responsável pela extração das características numéricas e armazenamento em formato CSV para o treinamento.

preparar_dataset.py: Organiza o conjunto de dados bruto na estrutura de diretórios necessária, realizando a divisão entre treino, validação e teste.

app.html: Interface do usuário com design minimalista em tema escuro, permitindo o envio de imagens e a visualização dos relatórios de análise.

requirements.txt: Especificação das dependências e bibliotecas necessárias para a execução do sistema.

.env: Arquivo de configuração para chaves de API, mantendo a segurança das credenciais.

kingambit.png: Elemento gráfico de identificação visual do sistema.

sobre.pdf: Manual do usuário com instruções detalhadas de operação.

respostas.xlsx: Registro de avaliações enviadas pelos usuários.

modelo_salvo.pth: Estado persistido do modelo treinado.

normalizador.joblib: Objeto de escalonamento estatístico utilizado para manter a consistência dos dados de entrada.

CONJUNTO DE DADOS (DATASET)

O treinamento utilizou o dataset NTIRE Robust AI-Generated Image Detection. Diferente de abordagens anteriores focadas apenas em rostos, este conjunto abrange imagens generalistas de múltiplos geradores de IA.

A composição total é de 12.500 imagens únicas, distribuídas de forma equilibrada entre classes reais e sintéticas. Metade do conjunto apresenta distorções como compressão JPEG, desfoque e ruído, o que confere ao modelo maior resistência a imagens que sofreram processamento prévio.

A divisão dos dados foi estabelecida em 70% para treinamento, 15% para validação e 15% para testes.

TECNOLOGIAS UTILIZADAS

O projeto foi construído utilizando as seguintes ferramentas:
Python 3.14 como linguagem base.
Flask para a camada de serviço e API.
PyTorch para o desenvolvimento e execução das redes neurais, com suporte a processamento via GPU.
EfficientNet-B0 para a extração de padrões visuais complexos.
OpenCV e Pillow para o processamento de imagens e filtros espaciais.
ExifTool para a inspeção de metadados ocultos.
SerpApi para a integração com serviços de busca externa.
Pandas e Scikit-learn para manipulação de dados e normalização estatística.

DETALHAMENTO DAS ANÁLISES

6.1 Extração de Metadados
O sistema utiliza o ExifTool para identificar informações sobre o dispositivo de captura, softwares de edição, coordenadas de GPS e datas de modificação. A ausência ou inconsistência desses dados é frequentemente um indicador de manipulação.

6.2 Análise de Nível de Erro (ELA)
Esta técnica identifica discrepâncias no nível de compressão JPEG. Ao recomprimir a imagem e comparar os pixels com o original, áreas editadas costumam apresentar padrões de brilho distintos, cujas médias e desvios padrão são enviados ao classificador.

6.3 Análise Forense Visual
Inclui o Mapa de Gradiente para detectar transições artificiais, o Mapa de Ruido para identificar inconsistências na textura do sensor e a Transformada de Fourier para revelar periodicidades sintéticas. Também são analisadas a Correlação RGB e a Aberração Cromática, buscando imperfeições naturais que geradores de IA muitas vezes falham em reproduzir.

MODELO DE APRENDIZAGEM DE MÁQUINA

O modelo utiliza uma arquitetura híbrida. Uma rede neural convolucional (EfficientNet-B0) extrai características visuais de alto nível, enquanto uma rede densa processa as 10 características forenses calculadas pelo sistema. A combinação desses dois vetores permite que o classificador tome uma decisão baseada tanto na estética da imagem quanto em suas propriedades estatísticas e físicas.

RESULTADOS E PERFORMANCE

O modelo atual apresenta uma acurácia de aproximadamente 78,46% no conjunto de testes. É importante observar que o recall (0,8401) é superior à precisão (0,7562), indicando que o sistema é conservador e tende a classificar imagens suspeitas como fakes para evitar falsos negativos, o que é preferível em contextos de segurança forense.

EXECUÇÃO DO SISTEMA

Para preparar os dados:
python preparar_dataset.py

Para extrair características:
python extrair_features.py

Para treinar o modelo:
python modelozudo.py treinar

Para iniciar o servidor:
python server.py

O sistema operará no endereço local na porta 5000.

LIMITAÇÕES

O sistema é focado em imagens estáticas e a precisão pode ser reduzida em arquivos de baixíssima resolução, onde a informação de pixel é insuficiente para uma análise estatística confiável. O resultado deve ser interpretado como um índice probabilístico e não como uma prova irrefutável.