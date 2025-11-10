from typing import List, Dict, Any, Optional , Tuple
import logging
from abc import ABC, abstractmethod
import torch
from sentence_transformers import CrossEncoder
from app.models.entities import SearchResult

logger = logging.getLogger(__name__)

class BaseRanker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass

class CrossEncoderRanker(BaseRanker):
    def __init__(self,
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: str = "auto",
                 batch_size: int = 32,
                 max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        self.device = self._get_device(device)

        logger.info(f"Loading CrossEncoder model '{self.model_name}' on device '{self.device}' fro re-ranking")
        self.model = self._load_model()

    def _get_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"  
            else:
                device = "cpu"
        return device
    
    def _load_model(self) -> CrossEncoder:
        try:
            model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
            logger.info(f"Reranker Model '{self.model_name}' loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Error loading Reranker model '{self.model_name}': {e}")
            raise e

    def rerank(self, query: str, 
               results: List[SearchResult],
               top_k: int = None,
               boost_exact_matches: bool = True,
               boost_value: float = 0.1) -> List[SearchResult]:
        if not results:
            return results
        try:
            pairs = []
            for result in results:
                text = result.chunk_text
                if len(text) > self.max_length * 4:
                    text = text[:self.max_length * 4]
                pairs.append((query, text))
            
            logger.info(f"Reranking {len(results)} results for query: '{query}'")
            scores = self.model.predict(pairs, batch_size=self.batch_size)
            reranked_results = []
            for idx, result in enumerate(results):
                rerank_score = float(scores[idx])
                if boost_exact_matches and result.exact_match_score > 0:
                    rerank_score += boost_value
                reranked_result = SearchResult(
                    chunk_id=result.chunk_id,
                    chunk_text=result.chunk_text,
                    file_name=result.file_name,
                    file_path=result.file_path,
                    file_type=result.file_type,
                    chunk_index=result.chunk_index,
                    total_chunks=result.total_chunks,
                    semantic_score=result.semantic_score,
                    keyword_score=result.keyword_score,
                    exact_match_score=result.exact_match_score,
                    final_score=result.final_score,
                    rerank_score=rerank_score,       
                    score=rerank_score,             
                    metadata={
                        **result.metadata,
                        'original_rrf_score': result.final_score,
                        'reranker_model': self.model_name
                    }
                )
                reranked_results.append(reranked_result)
            reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)
            return reranked_results[:top_k] if top_k is not None else reranked_results
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            raise e
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "model_type": "CrossEncoder"
        }

class HybridReranker(BaseRanker):
    def __init__(self,
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 cross_encoder_weight: float = 0.7,
                 rrf_weight: float = 0.3,
                 exact_match_weight: float = 0.1,
                 device: str = "auto"):
        self.cross_encoder_name = cross_encoder_model
        self.cross_encoder_weight = cross_encoder_weight
        self.rrf_weight = rrf_weight
        self.exact_match_weight = exact_match_weight
        self.cross_encoder = CrossEncoderRanker(self.cross_encoder_name, device=device)

        self._normalize_weights()
    
    def _get_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"  
            else:
                device = "cpu"
        return device
    
    def _normalize_weights(self):
        total = self.cross_encoder_weight + self.rrf_weight + self.exact_match_weight
        self.cross_encoder_weight /= total
        self.rrf_weight /= total
        self.exact_match_weight /= total
    
    def rerank(self, query: str, results: List[SearchResult], top_k: int = None) -> List[SearchResult]:
        if not results:
            return results
        ce_results = self.cross_encoder.rerank(query, results, top_k=None, boost_exact_matches=False)

        if ce_results:
            max_ce_score = max(r.rerank_score for r in ce_results)
            min_ce_score = min(r.rerank_score for r in ce_results)
            ce_range = max_ce_score - min_ce_score if max_ce_score != min_ce_score else 1.0

            max_rrf_score = max(r.final_score for r in results)
            min_rrf_score = min(r.final_score for r in results)
            rrf_range = max_rrf_score - min_rrf_score if max_rrf_score != min_rrf_score else 1.0
        
        hybrid_results = []
        for result in ce_results:
            norm_ce_score = (result.rerank_score - min_ce_score) / ce_range 
            norm_rrf_score = (result.final_score - min_rrf_score) / rrf_range
            exact_bonus = 1.0 if result.exact_match_score > 0 else 0.0

            hybrid_score = (
                self.cross_encoder_weight * norm_ce_score +
                self.rrf_weight * norm_rrf_score +
                self.exact_match_weight * exact_bonus
            )
            hybrid_result = SearchResult(
                chunk_id=result.chunk_id,
                chunk_text=result.chunk_text,
                file_name=result.file_name,
                file_path=result.file_path,
                file_type=result.file_type,
                chunk_index=result.chunk_index,
                total_chunks=result.total_chunks,
                semantic_score=result.semantic_score,
                keyword_score=result.keyword_score,
                exact_match_score=result.exact_match_score,
                final_score=result.final_score,
                rerank_score=result.rerank_score,
                score=hybrid_score,  
                metadata={
                    **result.metadata,
                    'hybrid_score': hybrid_score,
                    'normalized_ce_score': norm_ce_score,
                    'normalized_rrf_score': norm_rrf_score,
                    'exact_match_bonus': exact_bonus,
                    'reranker_type': 'hybrid'
                }
            )
            hybrid_results.append(hybrid_result)
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        return hybrid_results[:top_k] if top_k is not None else hybrid_results
    
    def get_model_info(self) -> Dict[str, Any]:
        ce_info = self.cross_encoder.get_model_info()
        return {
            **ce_info,
            "model_type": "hybrid",
            "weights": {
                "cross_encoder": self.cross_encoder_weight,
                "rrf": self.rrf_weight,
                "exact_match": self.exact_match_weight
            }
        }