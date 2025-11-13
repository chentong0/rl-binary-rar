import glob
import json
import os
import pickle
import time
from ast import literal_eval
from typing import List, Tuple

import numpy as np
import requests
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel
import faiss


class LocalSearchAPI():
    def __init__(self, passages_path=None, passages_embeddings_path=None, model_name_or_path=None, lazy_loading=True, device='cpu'):
        # invariant variables
        self.passages_path = passages_path
        self.passages_embeddings_path = passages_embeddings_path
        self.model_name_or_path = model_name_or_path

        self.lazy_loading = lazy_loading
        self.device = device

        if not self.lazy_loading:
            self.load_index()


        # # cache related
        # self.cache_file = "data/cache/search_cache.json"
        # self.cache_dict = self.load_cache()
        # self.add_n = 0
        # self.save_interval = 10

    def load_index(self):
        self.search_api = FAVALocalSearchAPI(
            passages_path=self.passages_path,
            passages_embeddings_path=self.passages_embeddings_path,
            model_name_or_path=self.model_name_or_path,
            per_gpu_batch_size=64,
            # device='cuda' if torch.cuda.is_available() else 'cpu',
            device=self.device,
        )
    
    def unload_index(self):
        del self.search_api
        import gc

        import torch
        torch.cuda.empty_cache()
        gc.collect()

    def get_snippets(self, claim_lst, n_docs=10):

        if self.lazy_loading:
            self.load_index()

        text_claim_snippets_dict = {}

        results = self.search_api.search(claim_lst, n_docs)
        for i, (query, result) in enumerate(zip(claim_lst, results)):
            text_claim_snippets_dict[query] = [{
                "title": psg["title"],
                "snippet": psg["text"],
                "link": "",
            } for psg in result["passages"]]

        if self.lazy_loading:
            self.unload_index()

        return text_claim_snippets_dict

    # def save_cache(self):
    #     # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
    #     cache = self.load_cache().items()
    #     for k, v in cache:
    #         self.cache_dict[k] = v
    #     print(f"Saving search cache ...")
    #     with open(self.cache_file, "w") as f:
    #         json.dump(self.cache_dict, f, indent=4)

    # def load_cache(self):
    #     if os.path.exists(self.cache_file):
    #         with open(self.cache_file, "r") as f:
    #             # load a json file
    #             cache = json.load(f)
    #             print(f"Loading cache ...")
    #     else:
    #         cache = {}
    #     return cache

class FAVALocalSearchAPI():
    def __init__(self,
        passages_path,
        passages_embeddings_path,
        model_name_or_path=None,
        per_gpu_batch_size=64,
        no_fp16=False,
        question_maxlength=512,
        save_or_load_index=True,
        indexing_batch_size=2555904,
        projection_size=768,
        # n_subquantizers=0,
        # n_bits=8,
        n_list=None,
        m_hnsw=None,
        m_pq=None,
        # n_list=65536,
        # m_hnsw=32,
        # m_pq=32,
        device='cpu',
    ):
        self.passages_path = passages_path
        self.passages_embeddings_path = passages_embeddings_path
        self.model_name_or_path = model_name_or_path
        self.per_gpu_batch_size = per_gpu_batch_size
        self.no_fp16 = no_fp16
        self.question_maxlength = question_maxlength
        self.save_or_load_index = save_or_load_index
        self.indexing_batch_size = indexing_batch_size
        self.projection_size = projection_size
        # self.n_subquantizers = n_subquantizers
        # self.n_bits = n_bits
        self.n_list = n_list
        self.m_hnsw = m_hnsw
        self.m_pq = m_pq
        self.device = device

        self._load_encoder()
        self._load_index()
        self._load_passages()


    def _load_encoder(self):
        print(f"Loading model from: {self.model_name_or_path}")
        self.model = FAVAContriever.from_pretrained(self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if not self.no_fp16:
            self.model = self.model.half()
        self.model = self.model.cuda()
        self.model.eval()

    def _load_index(self):

        def add_embeddings(index, embeddings, ids, indexing_batch_size):
            end_idx = min(indexing_batch_size, embeddings.shape[0])
            ids_toadd = ids[:end_idx]
            embeddings_toadd = embeddings[:end_idx]
            ids = ids[end_idx:]
            embeddings = embeddings[end_idx:]
            index.index_data(ids_toadd, embeddings_toadd)
            return embeddings, ids

        def index_encoded_data(index, embedding_files, indexing_batch_size):
            assert len(embedding_files) > 0, "No embedding files found"

            allids = []
            allembeddings = np.array([])
            for i, file_path in enumerate(embedding_files):
                print(f"Loading file {file_path}")
                with open(file_path, "rb") as fin:
                    ids, embeddings = pickle.load(fin)

                allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
                allids.extend(ids)
                while allembeddings.shape[0] > indexing_batch_size:
                    allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

            while allembeddings.shape[0] > 0:
                allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

            print("Data indexing completed.")

        self.index = FAVAIndexer(self.projection_size, self.n_list, self.m_hnsw, self.m_pq)

        if self.device == 'cuda':
            raise NotImplementedError("GPU is not supported for local search")
        #     print("Moving index to GPU...")
        #     self.index.to_gpu()
        #     print("Index moved to GPU.")

        # index all passages
        input_paths = glob.glob(self.passages_embeddings_path)
        input_paths = sorted(input_paths)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        if self.save_or_load_index and os.path.exists(index_path) and self.device == 'cpu':
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing passages from files {input_paths}")
            start_time_indexing = time.time()
            index_encoded_data(self.index, input_paths, self.indexing_batch_size)
            print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
            if self.save_or_load_index and self.device == 'cpu':
                self.index.serialize(embeddings_dir)
        

    def _load_passages(self):
        # load passages
        # passages = src.data.load_passages(args.passages)

        path = self.passages_path
        if not os.path.exists(path):
            # logger.info(f"{path} does not exist")
            return
        # logger.info(f"Loading passages from: {path}")
        print(f"Loading passages from: {path}")
        passages = []
        with open(path) as fin:
            if path.endswith(".jsonl"):
                import json
                for k, line in enumerate(fin):
                    ex = json.loads(line)
                    passages.append(ex)
            else:
                import csv
                reader = csv.reader(fin, delimiter="\t")
                for k, row in enumerate(reader):
                    if not row[0] == "id":
                        ex = {"id": row[0], "title": row[2], "text": row[1]}
                        passages.append(ex)

        self.passage_id_map = {x["id"]: x for x in passages}
        print(f"Loaded {len(self.passage_id_map)} passages")

    def search(self, queries, n_docs):
        # questions_embedding = embed_queries(args, queries, model, tokenizer)

        embeddings = []
        for i in range(0, len(queries), self.per_gpu_batch_size):
            batch = queries[i : i + self.per_gpu_batch_size]

            encoded_batch = self.tokenizer.batch_encode_plus(
                batch,
                return_tensors="pt",
                max_length=self.question_maxlength,
                padding=True,
                truncation=True,
            )
            encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
            with torch.inference_mode():
                output = self.model(**encoded_batch)
            embeddings.append(output.cpu())

        embeddings = torch.cat(embeddings, dim=0)
        print(f"Questions embeddings shape: {embeddings.size()}")
        embeddings = embeddings.numpy()


        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(embeddings, n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        # add_passages(data, passage_id_map, top_ids_and_scores)
        results = []
        for i, results_and_scores in enumerate(top_ids_and_scores):
            docs = [self.passage_id_map[doc_id] for doc_id in results_and_scores[0]]
            scores = [float(score) for score in results_and_scores[1]]
            results.append({
                "passages": docs,
                "scores": scores,
            })
        return results


class FAVAContriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

class FAVAIndexer(object):

    def __init__(self, vector_sz, nlist=None, m_hnsw=None, m_pq=None):
        # if n_subquantizers > 0:
        #     self.index = faiss.IndexPQ(vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT)
        # else:
        #     self.index = faiss.IndexFlatIP(vector_sz)

        # nlist = 65536      # IVF clusters (IVF65536)
        # M_hnsw = 32        # Number of links for HNSW coarse quantizer
        # M_pq = 32          # Number of PQ segments (PQ32)

        # index = faiss.index_factory(d, "IVF65536_HNSW32,PQ32")
        if nlist is not None and m_hnsw is not None and m_pq is not None:
            self.index = faiss.index_factory(vector_sz, f"IVF{nlist}_HNSW{m_hnsw},PQ{m_pq}")
        else:
            assert nlist is None and m_hnsw is None and m_pq is None, "nlist, m_hnsw, and m_pq must be None if not all are provided"
            self.index = faiss.index_factory(vector_sz, "Flat")

        #self.index_id_to_db_id = np.empty((0), dtype=np.int64)
        self.index_id_to_db_id = []

    # def to_gpu(self):
    #     res = faiss.StandardGpuResources()  # use a single GPU
    #     co = faiss.GpuClonerOptions()
    #     co.useFloat16 = True
    #     co.useFloat16Scalar = True    # L2 distances also in f16
    #     gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, self.index, co)
    #     self.index = gpu_index_flat

        # ngpus = 4
        # resources = [faiss.StandardGpuResources() for i in range(ngpus)]
        # vres = faiss.loader.GpuResourcesVector()
        # vdev = faiss.loader.Int32Vector()
        # for i, res in zip(range(ngpus), resources):
        #     vdev.push_back(i)
        #     vres.push_back(res)
        # co = faiss.GpuClonerOptions()
        # # co.useFloat16 = True
        # gpu_index_flat = faiss.index_cpu_to_gpu_multiple(vres, vdev, self.index, co)

        # gpu_index_flat = faiss.index_cpu_to_all_gpus(self.index, ngpu=4)
        # self.index = gpu_index_flat
        
        # gpu_id = 0
        # cfg = faiss.GpuIndexFlatConfig()
        # cfg.device = gpu_id
        # cfg.useFloat16 = True               # store vectors in fp16
        # index_gpu = faiss.GpuIndexFlatL2(res, d, cfg)
        

    def index_data(self, ids, embeddings):
        self._update_id_mapping(ids)
        embeddings = embeddings.astype('float32')
        if not self.index.is_trained:
            print(f"Training index with {len(embeddings)} embeddings")
            self.index.train(embeddings)
            print(f"Index trained with {len(embeddings)} embeddings")
        self.index.add(embeddings)

        print(f'Total data indexed {len(self.index_id_to_db_id)}')

    def search_knn(self, query_vectors: np.array, top_docs: int, index_batch_size: int = 2048) -> List[Tuple[List[object], List[float]]]:
        query_vectors = query_vectors.astype('float32')
        result = []
        nbatch = (len(query_vectors)-1) // index_batch_size + 1
        for k in tqdm(range(nbatch)):
            start_idx = k*index_batch_size
            end_idx = min((k+1)*index_batch_size, len(query_vectors))
            q = query_vectors[start_idx: end_idx]
            scores, indexes = self.index.search(q, top_docs)
            # convert to external ids
            db_ids = [[str(self.index_id_to_db_id[i]) for i in query_top_idxs] for query_top_idxs in indexes]
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
        return result

    def serialize(self, dir_path):
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'passage_pos_id_map.pkl')
        print(f'Serializing index to {index_file}, meta data to {meta_file}')

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode='wb') as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, dir_path):
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'passage_pos_id_map.pkl')
        print(f'Loading index from {index_file}, meta data from {meta_file}')

        self.index = faiss.read_index(index_file)
        print(f'Loaded index of type {type(self.index)} and size {self.index.ntotal}')

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert len(
            self.index_id_to_db_id) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'

    def _update_id_mapping(self, db_ids: List):
        #new_ids = np.array(db_ids, dtype=np.int64)
        #self.index_id_to_db_id = np.concatenate((self.index_id_to_db_id, new_ids), axis=0)
        self.index_id_to_db_id.extend(db_ids)


if __name__ == "__main__":
    # Initialize the LocalSearchAPI
    search_api = LocalSearchAPI(
        passages_path="index/psgs_w100.tsv",
        passages_embeddings_path="index/wikipedia_embeddings/passages_*",
        model_name_or_path="facebook/contriever-msmarco",
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    # Define a sample query
    sample_queries = ["What is the capital of France?", "Who wrote 'To Kill a Mockingbird'?"]

    # Perform the search
    print("Performing search for sample queries...")
    st_time = time.time()
    results = search_api.get_snippets(sample_queries)
    et_time = time.time()
    print(f"Search completed in {et_time - st_time:.2f} seconds.")

    # Print the results
    for query, snippets in results.items():
        print(f"\nQuery: {query}")
        print("Results:")
        for snippet in snippets:
            print(f"  - Snippet: {snippet['snippet']}")
