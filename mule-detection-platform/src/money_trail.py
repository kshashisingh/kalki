import pandas as pd
import networkx as nx


def trace_funds(G, account, depth=4):
    if account not in G:
        return pd.DataFrame()
    tree = nx.bfs_tree(G, account, depth_limit=depth)
    rows = []
    for u, v in tree.edges():
        e = G[u][v]
        rows.append({
            "from": u, "to": v,
            "amount": e.get("amount", 0),
            "time": str(e.get("datetime", "")),
        })
    return pd.DataFrame(rows)


def find_intermediaries(G, bad_accounts, top_n=20):
    und = G.to_undirected()
    btw = nx.betweenness_centrality(G, k=min(500, len(G)))
    rows = []
    for n in G.nodes():
        dist = min(
            (nx.shortest_path_length(und, n, b)
             for b in bad_accounts if nx.has_path(und, n, b)),
            default=99
        )
        if dist <= 2:
            rows.append({
                "account": n,
                "betweenness": round(btw[n], 6),
                "dist_to_bad": dist,
                "in_degree": G.in_degree(n),
                "out_degree": G.out_degree(n),
            })
    df = pd.DataFrame(rows).sort_values("betweenness", ascending=False)
    return df.head(top_n)


def chain_evidence(G, path):
    lines = []
    for u, v in zip(path[:-1], path[1:]):
        e = G[u][v]
        lines.append(f"{u} --[{e.get('amount', 0):,.2f}]--> {v}")
    return "\n".join(lines)