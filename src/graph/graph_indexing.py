import pandas as pd

def index_graph(input_path, output_edges, output_nodes):
    print("Loading edges...")
    df = pd.read_csv(input_path)

    print("Creating user index...")
    users = pd.unique(df[["source", "target"]].values.ravel())

    user_to_id = {u: i for i, u in enumerate(users)}

    df["src"] = df["source"].map(user_to_id)
    df["dst"] = df["target"].map(user_to_id)

    print("Saving indexed edges...")
    df[["src", "dst", "weight"]].to_csv(output_edges, index=False)

    print("Saving node mapping...")
    nodes = pd.DataFrame(list(user_to_id.items()), columns=["user", "id"])
    nodes.to_csv(output_nodes, index=False)

    print("Done!")

if __name__ == "__main__":
    index_graph(
        "data/graph/edges.csv",
        "data/graph/edges_indexed.csv",
        "data/graph/nodes.csv"
    )