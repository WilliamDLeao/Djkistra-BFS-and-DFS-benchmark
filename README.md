# Algoritmos de Busca em Grafos

Um projeto em Python que demonstra as diferenças entre três algoritmos clássicos de busca em grafos: BFS, DFS e Dijkstra.

## Descrição

Este código compara o comportamento de três algoritmos de busca para encontrar caminhos entre pontos em um grafo:

- **BFS (Busca em Largura)**: Encontra o caminho com menor número de arestas
- **DFS (Busca em Profundidade)**: Encontra um caminho (não necessariamente o mais curto)
- **Dijkstra**: Encontra o caminho de menor custo total considerando pesos nas arestas

## Como Executar

### Pré-requisitos

- Python 3.6+
- Bibliotecas: matplotlib, networkx

### Instalação das Dependências

```bash
pip install matplotlib networkx
```

### Execução

```bash
python algoritmo_busca.py
```

## Saída Esperada

O programa irá:

1. Mostrar todos os caminhos possíveis e seus custos(no terminal)
2. Executar os três algoritmos (BFS, DFS, Dijkstra)
3. Exibir os caminhos encontrados por cada algoritmo(no terminal)
4. Comparar resultados (comprimento, custo, tempo)
5. Gerar uma visualização gráfica do grafo com os caminhos destacados
