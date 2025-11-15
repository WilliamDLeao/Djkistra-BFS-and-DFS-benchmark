import heapq
import time
from collections import deque
import matplotlib.pyplot as plt
import networkx as nx

class GraphSearch:
    def __init__(self, graph):
        self.graph = graph
    
    def bfs_path(self, start, end):
        """Busca em Largura (BFS) - encontra o caminho mais curto em número de arestas"""
        if start == end:
            return [start]
            
        queue = deque([[start]])
        visited = set([start])
        
        while queue:
            path = queue.popleft()
            node = path[-1]
            
            if node == end:
                return path
            
            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
        
        return None
    
    def dfs_path(self, start, end):
        """Busca em Profundidade (DFS) - encontra um caminho, não necessariamente o mais curto"""
        if start == end:
            return [start]
            
        stack = [[start]]
        visited = set([start])
        
        while stack:
            path = stack.pop()
            node = path[-1]
            
            if node == end:
                return path
            
            for neighbor in reversed(self.graph.get(node, [])):  # reversed para ordem consistente
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    stack.append(new_path)
        
        return None
    
    def dijkstra_path(self, start, end, weights):
        """Algoritmo de Dijkstra - encontra o caminho de menor custo"""
        if start == end:
            return [start]
            
        # Inicialização correta - apenas o nó inicial com distância 0
        distances = {start: 0}
        previous = {}
        
        # Fila de prioridade com (distância, nó)
        priority_queue = [(0, start)]
        
        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            
            # Se chegamos ao destino, podemos parar
            if current_node == end:
                break
            
            # Se a distância atual é maior que a registrada, pule
            if current_distance > distances.get(current_node, float('inf')):
                continue
            
            # Explorar vizinhos
            for neighbor in self.graph.get(current_node, []):
                # Obter peso da aresta
                edge_weight = weights.get((current_node, neighbor), 1)
                new_distance = current_distance + edge_weight
                
                # Se encontramos um caminho melhor para o vizinho
                if new_distance < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_distance, neighbor))
        
        # Reconstruir o caminho se existe um para o destino
        if end not in previous and end != start:
            return None
            
        # Reconstruir caminho do fim para o início
        path = []
        current = end
        
        while current is not None:
            path.append(current)
            current = previous.get(current)
        
        # Inverter o caminho
        path.reverse()
        
        # Verificar se o caminho começa no start
        return path if path and path[0] == start else None

def create_sample_graph():
    """Cria um grafo de exemplo para demonstração"""
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B', 'G'],
        'E': ['B', 'F', 'H'],
        'F': ['C', 'E', 'I'],
        'G': ['D', 'H'],
        'H': ['E', 'G', 'I'],
        'I': ['F', 'H', 'J'],
        'J': ['I']
    }
    
    # Pesos para Dijkstra (alguns caminhos são mais longos)
    weights = {
        ('A', 'B'): 1, ('B', 'A'): 1,
        ('A', 'C'): 4, ('C', 'A'): 4,
        ('B', 'D'): 1, ('D', 'B'): 1,
        ('B', 'E'): 2, ('E', 'B'): 2,
        ('C', 'F'): 1, ('F', 'C'): 1,
        ('D', 'G'): 3, ('G', 'D'): 3,
        ('E', 'F'): 1, ('F', 'E'): 1,
        ('E', 'H'): 2, ('H', 'E'): 2,
        ('F', 'I'): 3, ('I', 'F'): 3,
        ('G', 'H'): 1, ('H', 'G'): 1,
        ('H', 'I'): 1, ('I', 'H'): 1,
        ('I', 'J'): 2, ('J', 'I'): 2
    }
    
    return graph, weights

def visualize_graph(graph, weights, paths, title):
    """Visualiza o grafo e os caminhos encontrados"""
    G = nx.Graph()
    
    # Adicionar nós e arestas
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            weight = weights.get((node, neighbor), 1)
            G.add_edge(node, neighbor, weight=weight)
    
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(12, 8))
    
    # Desenhar o grafo
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos)
    
    # Desenhar arestas com pesos
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edges(G, pos, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Destacar diferentes caminhos com cores
    colors = ['red', 'blue', 'green']
    labels = ['BFS', 'DFS', 'Dijkstra']
    
    for i, (algorithm, path) in enumerate(paths.items()):
        if path:
            # Desenhar arestas do caminho
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                                 edge_color=colors[i], width=3, alpha=0.7)
            # Destacar nós do caminho
            nx.draw_networkx_nodes(G, pos, nodelist=path, 
                                 node_color=colors[i], node_size=700)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def debug_dijkstra(graph, weights, start, end):
    """Função para debug do Dijkstra"""
    print(f"\n🔍 DEBUG DIJKSTRA: {start} → {end}")
    print("-" * 40)
    
    # Mostrar todos os caminhos possíveis e seus custos
    def find_all_paths(current, end, path, visited, all_paths):
        if current == end:
            all_paths.append(path[:])
            return
        
        visited.add(current)
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                path.append(neighbor)
                find_all_paths(neighbor, end, path, visited, all_paths)
                path.pop()
        visited.remove(current)
    
    all_paths = []
    find_all_paths(start, end, [start], set(), all_paths)
    
    print("Todos os caminhos possíveis:")
    for path in all_paths:
        cost = sum(weights.get((path[i], path[i+1]), 1) for i in range(len(path)-1))
        print(f"  {' → '.join(path)} (custo: {cost})")
    
    # Encontrar o caminho de menor custo manualmente
    if all_paths:
        best_path = min(all_paths, key=lambda p: sum(weights.get((p[i], p[i+1]), 1) for i in range(len(p)-1)))
        best_cost = sum(weights.get((best_path[i], best_path[i+1]), 1) for i in range(len(best_path)-1))
        print(f"\n🎯 Caminho ótimo esperado: {' → '.join(best_path)} (custo: {best_cost})")

def compare_algorithms():
    """Compara os três algoritmos em um exemplo prático"""
    graph, weights = create_sample_graph()
    searcher = GraphSearch(graph)
    
    start, end = 'A', 'J'
    
    print("=" * 60)
    print(f"COMPARAÇÃO DE ALGORITMOS: {start} → {end}")
    print("=" * 60)
    
    # Debug: mostrar caminho ótimo esperado
    debug_dijkstra(graph, weights, start, end)
    
    # Executar BFS
    start_time = time.time()
    bfs_path = searcher.bfs_path(start, end)
    bfs_time = time.time() - start_time
    
    # Executar DFS
    start_time = time.time()
    dfs_path = searcher.dfs_path(start, end)
    dfs_time = time.time() - start_time
    
    # Executar Dijkstra
    start_time = time.time()
    dijkstra_path = searcher.dijkstra_path(start, end, weights)
    dijkstra_time = time.time() - start_time
    
    # Calcular custos
    def calculate_cost(path, weights):
        if not path:
            return float('inf')
        cost = 0
        for i in range(len(path) - 1):
            cost += weights.get((path[i], path[i+1]), 1)
        return cost
    
    bfs_cost = calculate_cost(bfs_path, weights)
    dfs_cost = calculate_cost(dfs_path, weights)
    dijkstra_cost = calculate_cost(dijkstra_path, weights)
    
    # Resultados
    results = {
        'BFS': {
            'path': bfs_path,
            'time': bfs_time,
            'cost': bfs_cost,
            'length': len(bfs_path) if bfs_path else 0
        },
        'DFS': {
            'path': dfs_path,
            'time': dfs_time,
            'cost': dfs_cost,
            'length': len(dfs_path) if dfs_path else 0
        },
        'Dijkstra': {
            'path': dijkstra_path,
            'time': dijkstra_time,
            'cost': dijkstra_cost,
            'length': len(dijkstra_path) if dijkstra_path else 0
        }
    }
    
    # Exibir resultados
    print(f"\n{'='*50}")
    print("RESULTADOS OBTIDOS:")
    print(f"{'='*50}")
    
    for algorithm, result in results.items():
        print(f"\n{algorithm}:")
        if result['path']:
            print(f"  Caminho: {' → '.join(result['path'])}")
            print(f"  Comprimento: {result['length']} nós")
            print(f"  Custo total: {result['cost']}")
            print(f"  Tempo: {result['time']:.6f} segundos")
        else:
            print(f"  ❌ Nenhum caminho encontrado")
            print(f"  Tempo: {result['time']:.6f} segundos")
    
    # Comparação
    successful_results = {k: v for k, v in results.items() if v['path']}
    if successful_results:
        print(f"\n{'='*40}")
        print("ANÁLISE COMPARATIVA:")
        print(f"{'='*40}")
        print(f"Menor caminho (nós): {min(successful_results.items(), key=lambda x: x[1]['length'])[0]}")
        print(f"Menor custo: {min(successful_results.items(), key=lambda x: x[1]['cost'])[0]}")
        print(f"Mais rápido: {min(results.items(), key=lambda x: x[1]['time'])[0]}")
    
    # Visualizar
    paths_for_viz = {
        'BFS': bfs_path,
        'DFS': dfs_path,
        'Dijkstra': dijkstra_path
    }
    visualize_graph(graph, weights, paths_for_viz, 
                   f"Comparação de Algoritmos: {start} → {end}")

def explain_differences():
    """Explica as diferenças entre os algoritmos"""
    print("\n" + "="*60)
    print("EXPLICAÇÃO DAS DIFERENÇAS:")
    print("="*60)
    
    print("\n🔍 BFS (Busca em Largura):")
    print("  • Explora todos os vizinhos primeiro")
    print("  • Garante o caminho mais curto em número de arestas")
    print("  • Usa fila (FIFO)")
    print("  • Complexidade: O(V + E)")
    
    print("\n🔍 DFS (Busca em Profundidade):")
    print("  • Explora o máximo possível em uma direção antes de voltar")
    print("  • Não garante o caminho mais curto")
    print("  • Pode ser mais rápido se o alvo estiver profundo")
    print("  • Usa pilha (LIFO)")
    print("  • Complexidade: O(V + E)")
    
    print("\n📊 Dijkstra:")
    print("  • Considera pesos/custos nas arestas")
    print("  • Garante o caminho de menor custo total")
    print("  • Usa fila de prioridade (min-heap)")
    print("  • Complexidade: O((V + E) log V)")
    print("  • Ideal para redes com diferentes custos")

if __name__ == "__main__":
    compare_algorithms()
    explain_differences()