import pandas as pd
import numpy as np
import networkx as nx
import gc

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False


def build_graph_features(txns, bad_accounts=None):
    """
    FAST VERSION — Sampled betweenness, efficient graph ops
    """
    print("  Building graph features (fast mode)...")

    # ── Graph banao ──
    print("    Creating graph...")
    G = nx.from_pandas_edgelist(
        txns, "src", "dst",
        edge_attr=["amount", "datetime"],
        create_using=nx.DiGraph
    )
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"    Graph: {n_nodes:,} nodes, {n_edges:,} edges")

    # ── Centralities (FAST — limited samples) ──
    print("    Computing centralities...")

    # PageRank — fast algorithm
    pr = nx.pagerank(G, weight="amount", max_iter=30, tol=1e-3)

    # Betweenness — SAMPLED (bahut fast hoga)
    sample_size = min(200, max(n_nodes - 1, 1))
    btw = nx.betweenness_centrality(G, k=sample_size, seed=42)

    # Core number
    core = nx.core_number(G.to_undirected())

    # Degree
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    print("    Computing communities...")

    # ── Community (FAST) ──
    if HAS_LOUVAIN:
        comm = community_louvain.best_partition(
            G.to_undirected(),
            random_state=42,
            resolution=1.0,
        )
    else:
        comm = {n: 0 for n in G.nodes()}

    # ── Build DataFrame ──
    print("    Building feature table...")

    nodes = list(G.nodes())
    feats = pd.DataFrame({
        "acct":         nodes,
        "pagerank":     [pr.get(n, 0) for n in nodes],
        "betweenness":  [btw.get(n, 0) for n in nodes],
        "kcore":        [core.get(n, 0) for n in nodes],
        "in_degree":    [in_deg.get(n, 0) for n in nodes],
        "out_degree":   [out_deg.get(n, 0) for n in nodes],
        "community_id": [comm.get(n, -1) for n in nodes],
    })

    # ── Community bad rate + dist to bad ──
    if bad_accounts is not None and len(bad_accounts) > 0:
        print("    Computing community risk + distance to bad...")

        bad_set = set(str(b) for b in bad_accounts)

        # Community bad rate (FAST)
        node_is_bad = pd.Series(
            {n: int(str(n) in bad_set) for n in nodes}
        )
        node_comm = pd.Series(comm)
        comm_bad_rate = (
            node_is_bad.groupby(node_comm).mean()
            .rename("comm_bad_rate")
        )
        feats["comm_bad_rate"] = (
            feats.community_id.map(comm_bad_rate).fillna(0)
        )

        # Distance to bad (FAST — limited BFS)
        print("    Computing distance to bad accounts...")
        und = G.to_undirected()
        bad_list = list(bad_set)[:50]  # Top 50 bad accounts se dist check

        # BFS from each bad account (bulk approach)
        all_dists = {n: 99 for n in nodes}
        for bad in bad_list:
            if bad in und:
                dists = nx.single_source_shortest_path_length(
                    und, bad, cutoff=3
                )
                for node, d in dists.items():
                    if d < all_dists.get(node, 99):
                        all_dists[node] = d

        feats["dist_to_bad"] = [min(all_dists.get(n, 99), 10) for n in nodes]

    print(f"  ✅ Graph features: {feats.shape}")
    gc.collect()
    return feats, G