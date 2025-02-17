This folder contains plots of memory usage vs. time during indexing of 100k, 250k, 500k, 1M and 2M docs using raw ColBERT and RAGatouille.

The plots in this folder are run using the following scripts.

RAGatouille (via Jupyter Notebook):

```python
def memory_monitor(stop_event, readings):
    while not stop_event.is_set():
        mem = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
        readings.append((datetime.now(), mem))
        time.sleep(5)

def log_memory_during_index():
    stop_event = threading.Event()
    readings = []
    monitor_thread = threading.Thread(target=memory_monitor, args=(stop_event, readings))
    monitor_thread.start()
    
    try:
        index_path = RAG.index(
            index_name=f"{dataset_name}_index",
            collection=passages[:ndocs]["text"],
            document_ids=passages[:ndocs]["_id"]
        )
    finally:
        stop_event.set()
        monitor_thread.join()
    
    return index_path, readings

index_path, memory_readings = log_memory_during_index()
```

ColBERT (via terminal):

```python
nbits = 2  
ndocs = 100_000
dataset_name = "Genomics"
index_name = f'{dataset_name}.{nbits}bits'

passages = load_dataset("UKPLab/dapr", f"{dataset_name}-corpus", split="test")
checkpoint = 'answerdotai/answerai-colbert-small-v1'
config = ColBERTConfig(doc_maxlen=256, nbits=nbits, kmeans_niters=4, avoid_fork_if_possible=True)
indexer = Indexer(checkpoint=checkpoint, config=config)

def memory_monitor(stop_event, readings):
    while not stop_event.is_set():
        mem = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
        readings.append((datetime.now(), mem))
        time.sleep(5)

def log_memory_during_index():
    stop_event = threading.Event()
    readings = []
    monitor_thread = threading.Thread(target=memory_monitor, args=(stop_event, readings))
    monitor_thread.start()
    
    try:
        with Run().context(RunConfig(nranks=1, experiment='notebook')):
            index_path = indexer.index(name=index_name, collection=passages[:ndocs]["text"], overwrite=True)

    finally:
        stop_event.set()
        monitor_thread.join()
    
    return index_path, readings

index_path, memory_readings = log_memory_during_index()

start_time = memory_readings[0][0]
index=[(t - start_time).total_seconds() for t, _ in memory_readings]
pd.Series(
    [mem for _, mem in memory_readings],
    index=index
).plot(
    title='System RAM (100k docs)',
    xlabel='Time (sec)',
    ylabel='Memory (GB)'
)

plt.savefig('colbert_100k.png')
plt.close()
```
