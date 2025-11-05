from typing import List, Optional, Dict, Tuple
from collections import defaultdict, Counter
import math
import re

from embedding_manager import EmbeddingManager
from vector_storage import VectorStorage
from app.models.entities import SearchResult
class BM25Search:
    def __init__(self, k: float = 1.2, b: float = 0.75):
        self.k = k
        self.b = b
        self.documents = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0.0
        self.corpus_size = 0
        
    def fit(self, documents: List[str]):
        self.documents = documents
        self.corpus_size = len(documents)
        
        # Tokenize and calculate document frequencies
        word_counts = []
        df = defaultdict(int)  # Document frequency
        
        for doc in documents:
            words = self._tokenize(doc.lower())
            word_counts.append(Counter(words))
            self.doc_len.append(len(words))
            
            # Count documents containing each word
            for word in set(words):
                df[word] += 1
        
        self.doc_freqs = word_counts
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        
        # Calculate IDF for each word
        for word, freq in df.items():
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5))
    
    def score_query(self, query: str) -> List[float]:
        query_words = self._tokenize(query.lower())
        scores = []
        
        for i, doc_word_counts in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_len[i]
            
            for word in query_words:
                if word in doc_word_counts and word in self.idf:
                    tf = doc_word_counts[word]
                    idf = self.idf[word]
                    
                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                    score += idf * (numerator / denominator)
            
            scores.append(score)
        
        return scores
    
    def _tokenize(self, text: str) -> List[str]:
        pattern = re.sub(r'[^\w\s]', '', text)
        return [word for word in pattern.split() if len(word) > 2]

class ExactMatchSearch:
    def score_exact_matches(self, query: str, documents: List[str]) -> List[float]:
        scores = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for doc in documents:
            doc_lower = doc.lower()
            score = 0.0
            
            # Exact phrase match (highest score)
            if query_lower in doc_lower:
                score += 10.0
            
            # All query words present (medium score)
            doc_words = set(doc_lower.split())
            if query_words.issubset(doc_words):
                score += 5.0
            
            # Partial word matches (lower score)
            word_matches = len(query_words.intersection(doc_words))
            if word_matches > 0:
                score += (word_matches / len(query_words)) * 2.0
            
            scores.append(score)
        
        return scores

class RRF:
    def __init__(self, k: int = 60):
        self.k = k
    
    def combine_rankings(self, 
                         semantic_results: List[Tuple[str, float]],
                         keyword_results: List[Tuple[str, float]], # (doc_id, score)
                         exact_results: Optional[List[Tuple[str, float]]], 
                         weights: Dict[str, float] = None) -> List[Tuple[str, float]]:
        if weights is None: weights = {'semantic': 1.0, 'keyword': 1.0, 'exact': 2.0}

        semantic_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(semantic_results)}
        keyword_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(keyword_results)}
        exact_ranks = {}

        if exact_results:
            exact_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(exact_results)}

        all_doc_ids = set(semantic_ranks.keys()) | set(keyword_ranks.keys()) | set(exact_ranks.keys())

        rrf_scores = {}
        for doc_id in all_doc_ids:
            score = 0.0
            if doc_id in semantic_ranks:
                score += weights['semantic'] / (self.k + semantic_ranks[doc_id])
            if doc_id in keyword_ranks:
                score += weights['keyword'] / (self.k + keyword_ranks[doc_id])
            if doc_id in exact_ranks:
                score += weights['exact'] / (self.k + exact_ranks[doc_id])
            rrf_scores[doc_id] = score
        
        combined_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return combined_results


class HybridSearch:
    def __init__(self, vector_storage: VectorStorage, embedding_manager: EmbeddingManager, 
                 bm25: BM25Search, exact_matcher: ExactMatchSearch, rrf_ranker: RRF):
        self.vector_storage = vector_storage
        self.embedding_manager = embedding_manager
        self.bm25 = bm25
        self.exact_matcher = exact_matcher
        self.rrf_ranker = rrf_ranker
        self.document_corpus = []
        self.doc_id_mapping = {}
        self.corpus_built = False
    
    def build_keyword_index(self, force_rebuild: bool = False):
        if self.corpus_built and not force_rebuild:
            return
        
        print("Building keyword search index...")
        
        try:
            all_points, _ = self.vector_storage.client.scroll(
                collection_name=self.vector_storage.collection_name,
                limit=10000,  
                with_payload=True
            )
            
            documents = []
            doc_mapping = {}
            
            for i, point in enumerate(all_points):
                if point.payload and 'chunk_text' in point.payload:
                    chunk_id = point.payload.get('original_chunk_id', str(point.id))
                    content = point.payload['chunk_text']
                    
                    documents.append(content)
                    doc_mapping[i] = {
                        'chunk_id': chunk_id,
                        'point_id': point.id,
                        'payload': point.payload
                    }
            
            self.bm25.fit(documents)
            self.document_corpus = documents
            self.doc_id_mapping = doc_mapping
            self.corpus_built = True
            
            print(f"Keyword index built with {len(documents)} documents")
        except Exception as e:
            print(f"Error building keyword index: {e}")
            raise e
            

    def hybrid_search(self, 
                      query: str, 
                      top_k: int = 5, 
                      semantic_weight: float = 1.0, 
                      keyword_weight: float = 1.0, 
                      exact_weight: float = 2.0,
                      search_filter: Optional[Dict] = None): 
        
        semantic_results = self._semantic_search(query, top_k * 3, search_filter)
        keyword_results = self._vector_search(query, top_k * 3)
        exact_results = self._exact_match_search(query, top_k * 2)

        weights = {
            'semantic': semantic_weight,
            'keyword': keyword_weight, 
            'exact': exact_weight
        }
        combined_results = self.rrf_ranker.combine_rankings(
            semantic_results,
            keyword_results,
            exact_results,
            weights
        )
        final_results = self._build_final_results(combined_results[:top_k], semantic_results, keyword_results, exact_results)
        return final_results

    def _semantic_search(self, query: str, top_k: int, search_filter: Optional[Dict]) -> List[dict]:
        try:
            query_embedding = self.embedding_manager.generate_query_embedding([query])

            results = self.vector_storage.search_query(
                collection_name=self.vector_storage.collection_name,
                query_embedding=query_embedding,
                top_k=top_k,
                query_filter=search_filter
            )
            
            return [(result.payload.get('original_chunk_id', str(result.id)), result.score) 
                   for result in results]
                   
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []

    def _keyword_search(self, query: str, top_k: int) -> List[dict]:
        try:
            scores = self.bm25.score_query(query)
        
            scored_docs = [(i, score) for i, score in enumerate(scores) if score > 0]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for i, score in scored_docs[:top_k]:
                if i in self.doc_id_mapping:
                    chunk_id = self.doc_id_mapping[i]['chunk_id']
                    results.append((chunk_id, score))
            
            return results
            
        except Exception as e:
            print(f"Keyword search error: {e}")
            return []

    def _exact_match_search(self, query: str, top_k: int) -> List[dict]:
        try:
            scores = self.exact_matcher.score_exact_matches(query, self.document_corpus)
            
            scored_docs = [(i, score) for i, score in enumerate(scores) if score > 0]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for i, score in scored_docs[:top_k]:
                if i in self.doc_id_mapping:
                    chunk_id = self.doc_id_mapping[i]['chunk_id']
                    results.append((chunk_id, score))
            
            return results
            
        except Exception as e:
            print(f"Exact match search error: {e}")
            return []

    def _build_final_results(self, 
                             combined_results: List[Tuple[str, float]],
                           semantic_results: List[Tuple[str, float]],
                           keyword_results: List[Tuple[str, float]], 
                           exact_results: List[Tuple[str, float]]) -> List[dict]:
        semantic_scores = dict(semantic_results)
        keyword_scores = dict(keyword_results)
        exact_scores = dict(exact_results)
        
        results = []

        for chunk_id, final_score in combined_results:
            doc_data = None
            for doc_info in self.doc_id_mapping.values():
                if doc_info['chunk_id'] == chunk_id:
                    doc_data = doc_info
                    break
            
            if doc_data:
                payload = doc_data['payload']
                
                result = SearchResult(
                    chunk_id=chunk_id,
                    content=payload.get('chunk_text', ''),
                    file_name=payload.get('file_name', ''),
                    file_path=payload.get('file_path', ''),
                    file_type=payload.get('file_type', ''),
                    chunk_index=payload.get('chunk_index', 0),
                    semantic_score=semantic_scores.get(chunk_id, 0.0),
                    keyword_score=keyword_scores.get(chunk_id, 0.0),
                    exact_match_score=exact_scores.get(chunk_id, 0.0),
                    final_score=final_score,
                    metadata=payload.get('metadata', {})
                )
                
                results.append(result)
        
        return results