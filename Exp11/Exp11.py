import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

edges = pd.read_csv("graph.csv")
heuristic_df = pd.read_csv("heuristic.csv")

if "cost" in edges.columns:
    edges["cost"] = pd.to_numeric(edges["cost"], errors="coerce").fillna(1)
    graph = edges.groupby("from").apply(
        lambda g: {row["to"]: row["cost"] for _, row in g.iterrows()}
    ).to_dict()
else:
    graph = edges.groupby("from")["to"].apply(list).to_dict()
    graph = {k: {v: 1 for v in vs} for k, vs in graph.items()}

    
heuristic = dict(zip(heuristic_df["node"], heuristic_df["heuristic"]))

for node in heuristic:
    graph.setdefault(node, {})

start, goal = "A", "G"
path, current = [start], start
path_cost = 0

while current != goal:
    neighbors = graph[current]
    if not neighbors:
        print("Dead end at:", current)
        break

    nxt = min(neighbors, key=lambda n: (heuristic[n], neighbors[n]))

    if heuristic[nxt] >= heuristic[current]:
        print("Stuck at local minimum at:", current)
        break

    path_cost += neighbors[nxt]
    current = nxt
    path.append(current)


G = nx.from_pandas_edgelist(
    edges, "from", "to", edge_attr=True, create_using=nx.DiGraph()
)

pos = nx.spring_layout(G, seed=42)

nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=1000, font_weight="bold")
nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels={(u, v): f"{d.get('cost', 1)}" for u, v, d in G.edges(data=True)},
)

nx.draw_networkx_labels(G, pos, labels={n: f"h={heuristic[n]}" for n in G}, font_color="red", verticalalignment="bottom")

nx.draw_networkx_edges(G, pos, edgelist=list(zip(path, path[1:])), edge_color="green", width=3)
nx.draw_networkx_nodes(G, pos, nodelist=path, node_color="limegreen")

print("Goal reached!" if path[-1] == goal else "Failed to reach the goal.")
print("Path taken:", path)
print("Total path cost:", path_cost)

plt.title("Hill Climbing Path Visualization (with Path Cost)")
plt.show()