# Desafio 1.2 - Classificação de Orientações de Partículas GroEL

## Descrição do Desafio

Após classificar e isolar partículas de interesse (como no [desafio 1.1]()), o próximo passo crítico no pipeline de Cryo-EM é determinar suas orientações relativas.  
Nosso desafio foi simular esse processo, classificando imagens projetadas da [proteína GroEL](https://en.wikipedia.org/wiki/GroEL) em 10 orientações distintas: `top`, `bottom`, `top-front`, `bottom-front`, `front`, `side`, `top-side`, `bottom-side`, `diagonal1` e `diagonal2`.  
>[!NOTE]
>A proteína GroEL é um dos complexos mais estudados da Cryo-EM.  
>É um chaperonina bacteriano com forma de barril, composta por 14 subunidades organizadas em dois anéis heptaméricos. GroEL é um modelo clássico em benchmark de SPA por ter simetria conhecida, alto contraste e uma forma bem definida, o que a torna ideal para testes de alinhamento e reconstrução.

A proteína GroEL, com sua simetria e forma bem definida, serviu como um modelo ideal para este teste, com as imagens simulando variações naturais de até `15 graus`. 

## Nossa estratégia

Decidimos abordar o desafio com um método similar ao que usamos inicialmente para o Desafio 1.1: o **Multi-scale Autoencoder**.  

A ideia foi treinar um modelo para capturar as características das projeções 2D, mesmo sem o uso de rótulos. 
Dessa forma, nosso modelo foi capaz de agrupar as imagens com base em suas orientações, que seriam as "classes" intrínsecas a serem descobertas.

O treinamento do Multi-scale Autoencoder foi executado por 100 épocas. Decidimos utilizar um multi-scale autoencoder, pois a arquitetura com "multi-scale" nos permite capturar tanto "detalhes finos" quanto "estruturas globais/maiores", sendo ideal para diferenciar as orientações sutis da proteína. 
> Esse é um exemplo da reconstrução das imagens originais utilizando nosso [multi-scale autencoder](https://github.com/felipevzps/LBB-4/blob/main/desafio-1/02/src/autoencoder.py).
>![Reconstrução Multiscale Autoencoder](https://github.com/felipevzps/LBB-4/blob/main/desafio-1/02/results/multiscale_autoencoder_reconstruction.png) 

Ao final da classificação, fizemos o mapeamento dos `clusters` para as `classes verdadeiras`.  
>Por exemplo, o `cluster 0` representa a proteína GroEL no angulo `top`.

## Análise dos Vetores Latentes: A Segregação das Poses da GroEL

A visualização t-SNE dos vetores latentes gerados pelo nosso Multi-scale Autoencoder revela a complexidade do problema de classificação de orientações. 
>![Classificação GroEL](https://github.com/felipevzps/LBB-4/blob/main/desafio-1/02/results/tSNE_latent_vectors.png)

O gráfico mostra que nosso modelo foi capaz de aprender e segregar os diferentes estados de orientação da proteína GroEL em clusters distintos, demonstrando a eficácia do autoencoder na captura das características geométricas das projeções 2D.

No entanto, a análise aprofundada dos clusters também aponta para uma limitação intrínseca da abordagem não supervisionada. Observamos que, mesmo no espaço latente, a separação entre as classes não é completa.  
Por exemplo, como evidenciado no `Cluster 2`, partículas que visualmente se assemelham a outras classes acabaram sendo agrupadas, indicando uma sobreposição de características entre as orientações projetadas.

Esse comportamento sugere que a tarefa de classificação de pose pura, onde cada imagem é mapeada a uma única e precisa orientação, é extremamente desafiadora. 
A proximidade de certas poses da GroEL no espaço 3D (como, por exemplo, as transições suaves de `top-front` para `front`) se reflete na proximidade de seus vetores latentes no gráfico t-SNE.

Apesar dessa análise visual validar a nossa hipótese de que a classificação não supervisionada tem um limite de acurácia, a transição para o uso de um multi-scale autoencoder ao invés de um simples autoencoder nos permitiu alcançar um ótimo resultado neste desafio!
