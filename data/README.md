Data Directory
This directory is a placeholder for datasets used by the xgCCA-SSG project. Datasets are downloaded automatically by the DGL and OGB libraries when running the code. No manual download is required.
Supported Datasets

OGB-BioKG: Biomedical knowledge graph for link prediction (downloaded via ogb.linkproppred).
Cora, Citeseer, Pubmed: Citation networks for node classification (downloaded via dgl.data).
Amazon Photo, Amazon Computers: Co-purchase networks (downloaded via dgl.data).
Coauthor CS, Coauthor Physics: Collaboration networks (downloaded via dgl.data).
OGBN-Arxiv: Citation network (downloaded via ogb.nodeproppred).
Reddit: Social network (downloaded via torch_geometric.datasets).

Notes

Datasets are stored in this directory by default.
Ensure write permissions for data/ when running the code.
Clear the directory to re-download datasets if needed.

