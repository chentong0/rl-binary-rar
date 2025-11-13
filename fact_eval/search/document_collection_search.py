from tqdm import tqdm
import multiprocessing as mp
from typing import List, Tuple, Any
import numpy as np
import time

class SearchEngine:
    def __init__(self, *args, **kwargs):
        pass

    def search(self, query_list, k=1):
        raise NotImplementedError


class SearchEngineDocumentCollection:
    def __init__(self, chunk_size=100, tokenizer_name=None, tokenizer_max_length=131072):
        self.embed_cache = {}
        self.add_n_embed = 0
        self.chunk_size = chunk_size

        if tokenizer_name is not None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.tokenizer_max_length = tokenizer_max_length
        else:
            self.tokenizer = None
            self.tokenizer_max_length = None

    def tokenizer_encode(self, text) -> List[int | str]:
        if self.tokenizer is None:
            return text.split()
        return self.tokenizer.encode(text, max_length=self.tokenizer_max_length, truncation=True)
    
    def tokenizer_decode(self, tokens) -> str:
        if self.tokenizer is None:
            return " ".join(tokens)
        return self.tokenizer.decode(tokens)
    
    def get_document_chunks(self, documents, chunk_size=100) -> Tuple[List[str], List[List[int | str]]]:
        document_chunks = []
        document_chunks_of_tokens = []
        for doc in documents:
            # Handle both string documents and dict documents with title/text
            if isinstance(doc, dict):
                # Extract text from document dict
                text = doc.get('text', '')
                title = doc.get('title', '')
                # Combine title and text for chunking
                full_text = f"{title} {text}".strip()
            else:
                # Fallback for string documents
                full_text = str(doc)
            
            tokenized_text = self.tokenizer_encode(full_text)
            chunks_of_tokens = [tokenized_text[i:i + chunk_size] for i in range(0, len(tokenized_text), chunk_size)]
            chunks = [self.tokenizer_decode(chunk) for chunk in chunks_of_tokens]
            document_chunks.extend(chunks)
            document_chunks_of_tokens.extend(chunks_of_tokens)
            # word_list = full_text.split()
            # chunks = [" ".join(word_list[i:i + chunk_size]) for i in range(0, len(word_list), chunk_size)]
            # document_chunks.extend(chunks)
        return tuple(document_chunks), tuple(document_chunks_of_tokens)

    def get_bm25_passages(self, documents, query, k):
        if len(documents) == 0:
            return []
        from fact_eval.utils.rank_bm25 import BM25Okapi
        doc_cache_key = hash(tuple((doc.get('title', ''), doc.get('text', '')) if isinstance(doc, dict) else doc for doc in documents))
        if doc_cache_key in self.embed_cache:
            document_chunks, bm25 = self.embed_cache[doc_cache_key]
        else:
            document_chunks, document_chunks_of_tokens = self.get_document_chunks(documents, chunk_size=self.chunk_size)
            bm25 = BM25Okapi(document_chunks_of_tokens)
            self.embed_cache[doc_cache_key] = (document_chunks, bm25)
            self.add_n_embed += 1
        tokenized_query = self.tokenizer_encode(query)
        scores = bm25.get_scores(tokenized_query)
        indices = np.argsort(-scores)[:k]
        return [document_chunks[i] for i in indices]

    def get_passages(self, documents, query, k):
        results = self.get_bm25_passages(documents, query, k)
        return results

    @staticmethod
    def _process_batch(args: Tuple[List[Any], List[str], int]) -> List[List[str]]:
        documents_list, query_list, k = args
        engine = SearchEngineDocumentCollection()
        results = []
        for documents, query in zip(documents_list, query_list):
            results.append(engine.get_passages(documents, query, k))
        return results

    def search(self, documents_list, query_list, k=1, num_processes=1, batch_size=8192):
        assert len(documents_list) == len(query_list), f"documents_list and query_list must have the same length, but got {len(documents_list)} and {len(query_list)}"

        if num_processes <= 1 or len(documents_list) < batch_size:
            # Use original sequential processing
            results = []
            for i, (documents, query) in enumerate(tqdm(zip(documents_list, query_list), total=len(documents_list), desc="Searching")):
                results.append(self.get_passages(documents, query, k))
            return results

        # Split work into batches for parallel processing

        batches = []
        for i in range(0, len(documents_list), batch_size):
            batch_docs = documents_list[i:i + batch_size]
            batch_queries = query_list[i:i + batch_size]
            batches.append((batch_docs, batch_queries, k))

        # Process batches in parallel
        with mp.Pool(processes=num_processes) as pool:
            batch_results = list(tqdm(
                pool.imap(self._process_batch, batches),
                total=len(batches),
                desc="Searching in documents"
            ))

        # Combine results from all batches
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)

        return results


# class DocDB(object):
#     """Sqlite backed document storage.

#     Implements get_doc_text(doc_id).
#     """
#     SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
#     MAX_LENGTH = 256

#     def __init__(self, db_path=None, data_path=None):
#         self.db_path = db_path
#         self.connection = sqlite3.connect(self.db_path, check_same_thread=False)

#         cursor = self.connection.cursor()
#         cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        
#         if len(cursor.fetchall())==0:
#             assert data_path is not None, f"{self.db_path} is empty. Specify `data_path` in order to create a DB."
#             print (f"{self.db_path} is empty. start building DB from {data_path}...")
#             self.build_db(self.db_path, data_path)

#     def __enter__(self):
#         return self

#     def __exit__(self, *args):
#         self.close()

#     def path(self):
#         """Return the path to the file that backs this database."""
#         return self.path

#     def close(self):
#         """Close the connection to the database."""
#         self.connection.close()

#     def build_db(self, db_path, data_path):
#         from transformers import RobertaTokenizer
#         tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        
#         titles = set()
#         output_lines = []
#         tot = 0
#         start_time = time.time()
#         c = self.connection.cursor()
#         c.execute("CREATE TABLE documents (title PRIMARY KEY, text);")

#         with open(data_path, "r") as f:
#             for line in f:
#                 dp = json.loads(line)
#                 title = dp["title"]
#                 text = dp["text"]
#                 if title in titles:
#                     continue
#                 titles.add(title)
#                 if type(text)==str:
#                     text = [text]
#                 passages = [[]]
#                 for sent_idx, sent in enumerate(text):
#                     assert len(sent.strip())>0
#                     tokens = tokenizer(sent)["input_ids"]
#                     max_length = MAX_LENGTH - len(passages[-1])
#                     if len(tokens) <= max_length:
#                         passages[-1].extend(tokens)
#                     else:
#                         passages[-1].extend(tokens[:max_length])
#                         offset = max_length
#                         while offset < len(tokens):
#                             passages.append(tokens[offset:offset+MAX_LENGTH])
#                             offset += MAX_LENGTH
                
#                 psgs = [tokenizer.decode(tokens) for tokens in passages if np.sum([t not in [0, 2] for t in tokens])>0]
#                 text = SPECIAL_SEPARATOR.join(psgs)
#                 output_lines.append((title, text))
#                 tot += 1

#                 if len(output_lines) == 1000000:
#                     c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
#                     output_lines = []
#                     print ("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time()-start_time)/60))

#         if len(output_lines) > 0:
#             c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
#             print ("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time()-start_time)/60))

#         self.connection.commit()
#         self.connection.close()

#     def get_text_from_title(self, title):
#         """Fetch the raw text of the doc for 'doc_id'."""
#         cursor = self.connection.cursor()
#         cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
#         results = cursor.fetchall()
#         results = [r for r in results]
#         cursor.close()
#         assert results is not None and len(results)==1, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
#         results = [{"title": title, "text": para} for para in results[0][0].split(SPECIAL_SEPARATOR)]
#         assert len(results)>0, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
#         return results

# class LocalSearchEngine(object):

#     def __init__(self, db, cache_path, embed_cache_path,
#                  retrieval_type="gtr-t5-large", batch_size=None):
#         self.db = db
#         self.cache_path = cache_path
#         self.embed_cache_path = embed_cache_path
#         self.retrieval_type = retrieval_type
#         self.batch_size = batch_size
#         assert retrieval_type=="bm25" or retrieval_type.startswith("gtr-")
        
#         self.encoder = None
#         self.load_cache()
#         self.add_n = 0
#         self.add_n_embed = 0

#     def load_encoder(self):
#         from sentence_transformers import SentenceTransformer
#         encoder = SentenceTransformer("sentence-transformers/" + self.retrieval_type)
#         encoder = encoder.cuda()
#         encoder = encoder.eval()
#         self.encoder = encoder
#         assert self.batch_size is not None
    
#     def load_cache(self):
#         if os.path.exists(self.cache_path):
#             with open(self.cache_path, "r") as f:
#                 self.cache = json.load(f)
#         else:
#             self.cache = {}
#         if os.path.exists(self.embed_cache_path):
#             with open(self.embed_cache_path, "rb") as f:
#                 self.embed_cache = pkl.load(f)
#         else:
#             self.embed_cache = {}
    
#     def save_cache(self):
#         if self.add_n > 0:
#             if os.path.exists(self.cache_path):
#                 with open(self.cache_path, "r") as f:
#                     new_cache = json.load(f)
#                 self.cache.update(new_cache)
            
#             with open(self.cache_path, "w") as f:
#                 json.dump(self.cache, f)
        
#         if self.add_n_embed > 0:
#             if os.path.exists(self.embed_cache_path):
#                 with open(self.embed_cache_path, "rb") as f:
#                     new_cache = pkl.load(f)
#                 self.embed_cache.update(new_cache)
            
#             with open(self.embed_cache_path, "wb") as f:
#                 pkl.dump(self.embed_cache, f)

#     def get_bm25_passages(self, topic, query, passages, k):
#         if topic in self.embed_cache:
#             bm25 = self.embed_cache[topic]
#         else:
#             bm25 = BM25Okapi([psg["text"].replace("<s>", "").replace("</s>", "").split() for psg in passages])
#             self.embed_cache[topic] = bm25
#             self.add_n_embed += 1
#         scores = bm25.get_scores(query.split())
#         indices = np.argsort(-scores)[:k]
#         return [passages[i] for i in indices]

#     def get_gtr_passages(self, topic, retrieval_query, passages, k):
#         if self.encoder is None:
#             self.load_encoder()
#         if topic in self.embed_cache:
#             passage_vectors = self.embed_cache[topic]
#         else:
#             inputs = [psg["title"] + " " + psg["text"].replace("<s>", "").replace("</s>", "") for psg in passages]
#             passage_vectors = self.encoder.encode(inputs, batch_size=self.batch_size, device=self.encoder.device)
#             self.embed_cache[topic] = passage_vectors
#             self.add_n_embed += 1
#         query_vectors = self.encoder.encode([retrieval_query], 
#                                             batch_size=self.batch_size,
#                                             device=self.encoder.device)[0]
#         scores = np.inner(query_vectors, passage_vectors)
#         indices = np.argsort(-scores)[:k]
#         return [passages[i] for i in indices]

#     def get_passages(self, topic, question, k):
#         retrieval_query = topic + " " + question.strip()
#         cache_key = topic + "#" + retrieval_query
        
#         if cache_key not in self.cache:
#             passages = self.db.get_text_from_title(topic)
#             if self.retrieval_type=="bm25":
#                 self.cache[cache_key] = self.get_bm25_passages(topic, retrieval_query, passages, k)
#             else:
#                 self.cache[cache_key] = self.get_gtr_passages(topic, retrieval_query, passages, k)
#             assert len(self.cache[cache_key]) in [k, len(passages)]
#             self.add_n += 1
        
#         return self.cache[cache_key]


# class GoogleSearchEngine:
#     def __init__(self):
#         # invariant variables
#         self.serper_key = os.getenv("SERPER_KEY_PRIVATE")
#         self.url = "https://google.serper.dev/search"
#         self.headers = {'X-API-KEY': self.serper_key,
#                         'Content-Type': 'application/json'}
#         # cache related
#         self.cache_file = "data/cache/search_cache.json"
#         self.cache_dict = self.load_cache()
#         self.add_n = 0
#         self.save_interval = 10

#     def get_snippets(self, claim_lst):
#         text_claim_snippets_dict = {}
#         for query in claim_lst:
#             search_result = self.get_search_res(query)
#             if "statusCode" in search_result:  # and search_result['statusCode'] == 403:
#                 print(search_result['message'])
#                 exit()
#             if "organic" in search_result:
#                 organic_res = search_result["organic"]
#             else:
#                 organic_res = []

#             search_res_lst = []
#             for item in organic_res:
#                 title = item["title"] if "title" in item else ""
#                 snippet = item["snippet"] if "snippet" in item else ""
#                 link = item["link"] if "link" in item else ""

#                 search_res_lst.append({"title": title,
#                                        "snippet": snippet,
#                                        "link": link})
#             text_claim_snippets_dict[query] = search_res_lst
#         return text_claim_snippets_dict

#     def get_search_res(self, query):
#         # check if prompt is in cache; if so, return from cache
#         cache_key = query.strip()
#         if cache_key in self.cache_dict:
#             # print("Getting search results from cache ...")
#             return self.cache_dict[cache_key]

#         payload = json.dumps({"q": query})
#         response = requests.request("POST",
#                                     self.url,
#                                     headers=self.headers,
#                                     data=payload)
#         response_json = literal_eval(response.text)

#         # update cache
#         self.cache_dict[query.strip()] = response_json
#         self.add_n += 1

#         # save cache every save_interval times
#         if self.add_n % self.save_interval == 0:
#             self.save_cache()

#         return response_json

#     def save_cache(self):
#         # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
#         cache = self.load_cache().items()
#         for k, v in cache:
#             self.cache_dict[k] = v
#         print(f"Saving search cache ...")
#         with open(self.cache_file, "w") as f:
#             json.dump(self.cache_dict, f, indent=4)

#     def load_cache(self):
#         if os.path.exists(self.cache_file):
#             with open(self.cache_file, "r") as f:
#                 # load a json file
#                 cache = json.load(f)
#                 print(f"Loading cache ...")
#         else:
#             cache = {}
#         return cache

