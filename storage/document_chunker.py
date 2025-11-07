from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from tiktoken import get_encoding, Encoding
from pathlib import Path
import re
import nltk
from collections import defaultdict, Counter
import spacy

from storage.file_handler import FileHandler

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    from nltk.tokenize import sent_tokenize, word_tokenize
except ImportError:
    print("NLTK not available. Using basic splitting.")
    def sent_tokenize(text):
        return re.split(r'[.!?]+', text)
    def word_tokenize(text):
        return text.split()


@dataclass   
class DocumentChunk:
    chunk_id: str
    file_path: str
    file_name: str
    file_type: str
    chunk_text: str
    chunk_index: int
    total_chunks: int
    file_hash: str
    modified_time: float
    metadata: Dict

class ContextEnrichedChunker:
    MAX_CHUNK_SIZE = 1024

    def __init__(self, 
                 target_chunk_size: int = 400,
                 min_chunk_size: int = 200,
                 max_chunk_size: int = 800,
                 context_window: int = 100,
                 overlap_ratio: float = 0.1,
                 file_handler: FileHandler = None,
                 tokenizer: Encoding = None):
        assert max_chunk_size <= self.MAX_CHUNK_SIZE, f"Max chunk size cannot exceed {self.MAX_CHUNK_SIZE} tokens."
        assert min_chunk_size < target_chunk_size < max_chunk_size, "Invalid chunk size relationship."

        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.context_window = context_window
        self.overlap_ratio = overlap_ratio
        self.file_handler = file_handler

        self.tokenizer = tokenizer or get_encoding("cl100k_base")
        self.spacy = None
        try:
            self.spacy = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy not available, using NLTK for processing")
    
    def chunk_text(self, text: str, file_path: Path) -> List[DocumentChunk]:
        if not text or len(text.strip()) == 0:
            return []
        file_hash = self.file_handler.compute_hash(str(file_path))
        stat = file_path.stat()

        doc_structure = self._analyze_document_structure(text)
        semantic_units = self._extract_semantic_units(text)
        raw_chunks = self._create_semantic_chunks(semantic_units, doc_structure)

        doc_chunks = []
        total_chunks = len(raw_chunks)

        for idx, chunk_data in enumerate(raw_chunks):
            if not chunk_data: continue
            token_count = len(self.tokenizer.encode(chunk_data["text"]))

            prev_context = self._extract_context_before(chunk_data["start_char"], text, self.context_window)
            next_context = self._extract_context_after(chunk_data["end_char"], text, self.context_window)

            section_header = self._find_section_header(chunk_data["start_char"], doc_structure["headers"])

            semantic_density = self._calculate_semantic_density(chunk_data['text'])
            readability_score = self._calculate_readability(chunk_data['text'])

            chunk_id = f"{file_hash}_{idx:04d}"

            enhanced_metadata = {
                "size_bytes": stat.st_size, # Original Metadata
                "indexed_at": datetime.now().isoformat(),
                "start_char": chunk_data['start_char'],
                "end_char": chunk_data['end_char'],
                "token_count": token_count,
                "context_before": prev_context, # Context enrichment
                "context_after": next_context,
                "section_header": section_header,
                "document_summary": doc_structure['summary'],
                "semantic_density": semantic_density, # Semantic analysis
                "readability_score": readability_score,
                "sentence_count": chunk_data['sentence_count'],
                "entity_mentions": chunk_data.get('entities', []),
                "topics": self._extract_topics(chunk_data['text']),
                "has_questions": chunk_data.get('has_questions', False), # Quality metrics
                "has_dialogue": chunk_data.get('has_dialogue', False),
                "word_count": chunk_data['word_count'],
            }

            doc_chunk = DocumentChunk(
                chunk_id=chunk_id,
                file_path=str(file_path.absolute()),
                file_name=file_path.name,
                file_type=file_path.suffix,
                chunk_index=idx,
                chunk_text=chunk_data['text'],
                total_chunks=total_chunks,
                file_hash=file_hash,
                modified_time=stat.st_mtime,
                metadata=enhanced_metadata
            )
            doc_chunks.append(doc_chunk)
        return doc_chunks
    
    def _analyze_document_structure(self, text: str) -> Dict[str, Any]:
        structure = {
            'headers': [],
            'sections': [],
            'paragraphs': [],
            'summary': '',
            'total_length': len(text),
            'estimated_reading_time': len(text.split()) / 200
        }
        lines = text.split('\n')
        current_section = None
        char_offset = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                char_offset += len(lines[i]) + 1  
                continue
            if self._is_likely_header(line):
                structure['headers'].append({
                    'text': line,
                    'line_number': i,
                    'char_position': char_offset,
                    'level': self._estimate_header_level(line)
                })
                current_section = line
            
            char_offset += len(lines[i]) + 1
        sentences = sent_tokenize(text)
        structure['summary'] = ' '.join(sentences[:3]) if len(sentences) >= 3 else text[:300]
        
        return structure

    def _is_likely_header(self, line: str) -> bool:
        if not line.strip():
            return False
            
        header_patterns = [
            r'^[A-Z][A-Z\s]{2,}$',        # ALL CAPS
            r'^Chapter\s+\d+',             # Chapter X
            r'^Section\s+\d+',             # Section X
            r'^\d+\.\s+[A-Z]',            # 1. Title
            r'^[IVX]+\.\s+[A-Z]',         # I. Roman numerals
        ]
        
        for pattern in header_patterns:
            if re.match(pattern, line.strip()):
                return True
                
        words = line.split()
        if (len(words) <= 8 and 
            len(line) < 80 and 
            line[0].isupper() and 
            not line.endswith('.') and
            sum(1 for w in words if w and w[0].isupper()) >= len(words) * 0.5):
            return True
            
        return False
    
    def _estimate_header_level(self, header: str) -> int:
        if re.match(r'^[A-Z][A-Z\s]{2,}$', header):
            return 1  # Major header
        elif re.match(r'^Chapter\s+\d+', header):
            return 1
        elif re.match(r'^\d+\.\s+', header):
            return 2
        else: return 3 # Minor header

    def _extract_semantic_units(self, text: str) -> List[Dict[str, Any]]:
        sentences = sent_tokenize(text)
        units = []
        char_offset = 0
        
        for idx, sentence in enumerate(sentences):
            start_pos = text.find(sentence, char_offset)
            if start_pos == -1:
                start_pos = char_offset
            
            end_pos = start_pos + len(sentence)
            char_offset = end_pos
            
            units.append({
                'text': sentence,
                'index': idx,
                'start_char': start_pos,
                'end_char': end_pos,
                'word_count': len(word_tokenize(sentence)),
                'is_question': sentence.strip().endswith('?'),
                'has_dialogue': '"' in sentence or "'" in sentence,
                'entities': self._extract_entities(sentence) if self.spacy else [],
            })
        
        return units

    def _extract_entities(self, text: str) -> List[str]:
        if not self.spacy: return []
        doc = self.spacy(text)
        return [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']]
    
    def _create_semantic_chunks(self, units: List[Dict], doc_structure: Dict) -> List[Dict[str, Any]]:
        chunks = []
        current_chunk = []
        current_size = 0
        
        for unit in units:
            unit_size = unit['word_count']
            potential_size = current_size + unit_size
            
            should_break = (
                (potential_size > self.max_chunk_size and current_chunk) or
                (potential_size > self.target_chunk_size and 
                 current_size >= self.min_chunk_size and 
                 self._is_good_break_point(unit))
            )
            
            if should_break:
                if current_chunk:
                    chunks.append(self._finalize_chunk(current_chunk))
                overlap_units = self._get_overlap_units(current_chunk)
                current_chunk = overlap_units + [unit]
                current_size = sum(u['word_count'] for u in current_chunk)
            else:
                current_chunk.append(unit)
                current_size += unit_size
        if current_chunk:
            chunks.append(self._finalize_chunk(current_chunk))
        
        return [chunk for chunk in chunks if chunk]  
    
    def _is_good_break_point(self, unit: Dict) -> bool:
        """Determine if this is a good place to break a chunk"""
        text = unit['text'].strip()
        
        # Good break points
        if text.endswith(('.', '!', '?')):
            return True
        if text.endswith(('."', '!"', '?"')):
            return True
        if unit.get('is_dialogue_end', False):
            return True
            
        return False
    
    def _get_overlap_units(self, chunk_units: List[Dict]) -> List[Dict]:
        if not chunk_units:
            return []
            
        total_words = sum(u['word_count'] for u in chunk_units)
        overlap_words = int(total_words * self.overlap_ratio)
        
        overlap_units = []
        word_count = 0
    
        for unit in reversed(chunk_units):
            if word_count + unit['word_count'] <= overlap_words:
                overlap_units.insert(0, unit)
                word_count += unit['word_count']
            else:
                break
                
        return overlap_units

    def _finalize_chunk(self, units: List[Dict]) -> Dict[str, Any]:
        if not units:
            return None
            
        text = ' '.join(unit['text'] for unit in units)
        
        return {
            'text': text,
            'units': units,
            'start_char': units[0]['start_char'],
            'end_char': units[-1]['end_char'],
            'word_count': sum(unit['word_count'] for unit in units),
            'sentence_count': len(units),
            'entities': list(set().union(*(unit.get('entities', []) for unit in units))),
            'has_questions': any(unit.get('is_question', False) for unit in units),
            'has_dialogue': any(unit.get('has_dialogue', False) for unit in units),
        }
    
    def _extract_context_before(self, start_char: int, full_text: str, window_size: int) -> str:
        context_start = max(0, start_char - window_size)
        context = full_text[context_start:start_char]
        sentences = sent_tokenize(context)
        if len(sentences) > 1:
            return ' '.join(sentences[1:]).strip()
        
        return context.strip()
    
    def _extract_context_after(self, end_char: int, full_text: str, window_size: int) -> str:
        context_end = min(len(full_text), end_char + window_size)
        context = full_text[end_char:context_end]
        sentences = sent_tokenize(context)
        if len(sentences) > 1:
            return ' '.join(sentences[:-1]).strip()
        
        return context.strip()
    
    def _find_section_header(self, char_position: int, headers: List[Dict]) -> str:
        relevant_header = ""
        for header in headers:
            if header['char_position'] <= char_position:
                relevant_header = header['text']
            else:
                break
                
        return relevant_header
    
    def _calculate_semantic_density(self, text: str) -> float:
        words = word_tokenize(text.lower())
        if not words:
            return 0.0
            
        unique_words = set(words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'have', 'has', 'had'}
        content_words = unique_words - stop_words
        density = len(content_words) / len(words) if words else 0.0
        return min(1.0, density)
    
    def _calculate_readability(self, text: str) -> float:
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        complexity = (avg_sentence_length / 20) + (avg_word_length / 10)
        readability = max(0.0, min(1.0, 1.0 - (complexity / 2)))
        
        return readability
    
    def _extract_topics(self, text: str) -> List[str]:
        words = word_tokenize(text.lower())
        word_freq = defaultdict(int)
        for word in words:
            if len(word) > 4 and word.isalpha():
                word_freq[word] += 1
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return [topic[0] for topic in topics if topic[1] > 1]

class DocumentChunker(ContextEnrichedChunker):
    def __init__(self, 
                 chunk_size: int = 450, 
                 overlap: int = 100, 
                 min_chunk_size: int = 200,
                 max_chunk_size: int = 700,
                 context_window: int = 100,
                 overlap_ratio: float = 0.15,
                 file_handler: FileHandler = None, 
                 tokenizer: Encoding = None):
        super().__init__(
            target_chunk_size=min(chunk_size, 512),  
            min_chunk_size=max(200, chunk_size // 2),
            max_chunk_size=min(700, chunk_size + 200),
            context_window=context_window,
            overlap_ratio=overlap_ratio,
            file_handler=file_handler,
            tokenizer=tokenizer
        )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, file_path: str) -> List[DocumentChunk]:
        path_object = Path(file_path)
        return super().chunk_text(text, path_object)
        # if not text or len(text.strip()) == 0:
        #     return []
        
        # chunks = []
        # chunk_id, start = 0, 0
        
        # file_hash = self.file_handler.compute_hash(file_path)
        # stat = file_path.stat()

        # tokens = self.tokenizer.encode(text)

        # while start < len(tokens):
        #     end = min(start + self.chunk_size, len(tokens))
        #     chunk_tokens = tokens[start:end]
        #     chunk_text = self.tokenizer.decode(chunk_tokens)
        #     chunk = {
        #         'chunk_id': chunk_id,
        #         'start_token': start,
        #         'end_token': end,
        #         'text': chunk_text,
        #         'token_length': len(chunk_text),
        #     }
        #     chunks.append(chunk)
        #     chunk_id += 1
        #     start += self.chunk_size - self.overlap
        
        # total_chunks = len(chunks)
        # doc_chunk_objects = []

        # for chunk in chunks:
        #     doc_chunk = DocumentChunk(
        #         chunk_id=f"{file_hash}_{chunk['chunk_id']}",
        #         file_path=str(file_path.absolute()),
        #         file_name=file_path.name,
        #         file_type=file_path.suffix,
        #         chunk_index=chunk['chunk_id'],
        #         chunk_text=chunk['text'],
        #         total_chunks=total_chunks,
        #         file_hash=file_hash,
        #         modified_time=stat.st_mtime,
        #         metadata={
        #             "size_bytes": stat.st_size,
        #             "indexed_at": datetime.now().isoformat(),
        #             "start_token": chunk['start_token'],
        #             "end_token": chunk['end_token'],
        #             "token_count": chunk['token_length']
        #         }
        #     )
        #     doc_chunk_objects.append(doc_chunk)
        # return doc_chunk_objects
