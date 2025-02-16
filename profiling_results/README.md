All runs are done on 1 GPU Jarvis Labs RTX6000Ada instance. `False` in the filename refers to `use_faiss=False`, `True` to `use_faiss_True`.

Here's the script for indexing using ColBERT:

```python
import colbert
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from datasets import load_dataset

def main():
    nbits = 2  
    ndocs = 100_000
    print(ndocs)
    dataset_name = "Genomics"
    index_name = f'{dataset_name}.{nbits}bits'

    passages = load_dataset("UKPLab/dapr", f"{dataset_name}-corpus", split="test")

    # checkpoint = 'colbert-ir/colbertv2.0'
    checkpoint = 'answerdotai/answerai-colbert-small-v1'

    with Run().context(RunConfig(nranks=1, experiment='notebook')):
        config = ColBERTConfig(doc_maxlen=256, nbits=nbits, kmeans_niters=4, avoid_fork_if_possible=True)
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=passages[:ndocs]["text"], overwrite=True)

if __name__ == '__main__':
    main()
```

and the script for RAGatouille:

```python
from memory_profiler import profile
from datasets import load_dataset
from ragatouille import RAGPretrainedModel

dataset_name = "Genomics"
passages = load_dataset("UKPLab/dapr", f"{dataset_name}-corpus", split="test")
RAG = RAGPretrainedModel.from_pretrained("answerdotai/answerai-colbert-small-v1")
ndocs=250_000

@profile
def _index(): 
    return RAG.index(
        index_name=f"{dataset_name}_index", 
        collection=passages[:ndocs]["text"], 
        document_ids=passages[:ndocs]["_id"],
        use_faiss=True # or False
    )

_index()
```

Here's the terminal command to run the scripts and profile them:

```bash
python -m memory_profiler ../colbert_index_2M.py > ../colbert_2M_RTX6000Ada.txt
```
