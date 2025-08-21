# Contributing to RAG Service 🤝

RAG Service에 기여해주셔서 감사합니다! 이 가이드는 지능형 질의응답 시스템 개발에 참여하는 방법을 설명합니다.

## 📋 기여 방식

### 🐛 버그 리포트
- 검색 정확도 문제, 답변 품질 이슈 등 구체적으로 신고
- 질문 예시, 기대 답변, 실제 답변 포함
- 사용된 문서 유형, 데이터베이스 상태 정보 제공

### 💡 기능 제안
- 새로운 검색 알고리즘, 언어 모델 통합 제안
- 답변 품질 개선, 인용 정확도 향상 아이디어
- 다국어 지원, 도메인 특화 기능 제안

### 🔧 코드 기여
1. Fork 생성
2. Feature branch 생성: `git checkout -b feature/HybridSearchEnhancement`
3. 변경사항 커밋: `git commit -m 'Improve vector search accuracy'`
4. Branch에 Push: `git push origin feature/HybridSearchEnhancement`
5. Pull Request 생성

## 🛠 개발 환경 설정

### 필요 조건
- Python 3.11+
- MongoDB Atlas 계정 (또는 로컬 MongoDB)
- Azure OpenAI 구독
- Docker (선택사항)

### 로컬 설정
```bash
# 저장소 클론
git clone https://github.com/your-username/rag-service.git
cd rag-service

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 환경변수 설정
cp env.example .env
# .env 파일에 MongoDB URI, Azure OpenAI 인증 정보 입력

# 개발 서버 실행
uvicorn main:app --reload --port 8002
```

### 개발용 MongoDB 설정
```bash
# Docker로 로컬 MongoDB 실행
docker run -d --name rag-mongodb \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=password \
  mongo:7.0

# 또는 MongoDB Atlas 사용 (권장)
# https://www.mongodb.com/atlas에서 무료 클러스터 생성
```

## 🧪 테스트

### 단위 테스트 실행
```bash
pytest tests/ -v
```

### 검색 품질 테스트
```bash
# 하이브리드 검색 정확도 테스트
python tests/test_search_quality.py

# RAG 파이프라인 end-to-end 테스트
python tests/test_rag_pipeline.py

# 답변 품질 평가
python tests/test_answer_quality.py
```

### 성능 벤치마크
```bash
# 검색 속도 벤치마크
python tests/benchmark_search.py

# 메모리 사용량 프로파일링
python -m memory_profiler tests/test_memory_usage.py

# 동시 요청 처리 테스트
python tests/test_concurrent_requests.py
```

## 🔍 RAG 시스템 개발 가이드

### 검색 성능 최적화
```python
# 좋은 예시: 효율적인 하이브리드 검색
async def hybrid_search(self, query: str, k: int = 6) -> List[Dict]:
    """하이브리드 검색 (텍스트 + 벡터)"""
    
    # 병렬 검색 실행
    text_task = self._atlas_text_search(query, k=20)
    vector_task = self._atlas_vector_search(query, k=20)
    
    text_results, vector_results = await asyncio.gather(text_task, vector_task)
    
    # RRF 융합으로 결과 결합
    return self._rrf_fuse(text_results, vector_results, topk=k)

# 나쁜 예시: 순차적 검색
def sequential_search(self, query: str):
    text_results = self._text_search(query)  # 동기 처리
    vector_results = self._vector_search(query)  # 동기 처리
    return text_results + vector_results  # 단순 결합
```

### 프롬프트 엔지니어링
```python
# 효과적인 RAG 프롬프트 설계
SYSTEM_PROMPT = """
너는 제공된 컨텍스트에서만 근거를 찾아 정확하고 유용한 답변을 제공한다.

지침:
1. 컨텍스트에 정보가 있으면 구체적으로 답변
2. 컨텍스트에 정보가 없으면 솔직히 모른다고 답변
3. 추측이나 일반적인 지식으로 답변하지 않음
4. 답변에 출처를 명시하지 않음 (별도 제공됨)

답변 형식:
- 명확하고 구조화된 설명
- 필요시 번호나 목록 활용
- 전문 용어는 쉽게 설명
"""

USER_PROMPT = """
질문: {question}

컨텍스트:
{context}

위 컨텍스트를 바탕으로 질문에 답변하세요.
"""
```

### 벡터 임베딩 최적화
```python
class OptimizedEmbedding:
    def __init__(self):
        self.embedding_cache = {}
        self.batch_size = 16
    
    @lru_cache(maxsize=1000)
    def embed_text(self, text: str) -> List[float]:
        """캐시된 임베딩 생성"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # 텍스트 전처리
        processed_text = self._preprocess_text(text)
        
        # 배치 처리로 효율성 향상
        embedding = self.embedding_model.embed_query(processed_text)
        
        self.embedding_cache[text] = embedding
        return embedding
    
    def _preprocess_text(self, text: str) -> str:
        """임베딩 품질 향상을 위한 전처리"""
        # 불필요한 공백, 특수문자 제거
        # 문장 정규화, 키워드 추출 등
        return processed_text
```

## 📏 코딩 스타일

### RAG 관련 코딩 규칙
```python
# 검색 결과 타입 힌트 사용
from typing import List, Dict, Optional, Tuple

SearchResult = Dict[str, Any]
DocumentChunk = Dict[str, str]
RetrievalScore = float

async def search_documents(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    max_results: int = 6
) -> List[SearchResult]:
    """문서 검색 함수"""
    pass

# 설정 값은 상수로 정의
class RAGConfig:
    DEFAULT_TOP_K = 6
    MAX_CONTEXT_LENGTH = 4000
    RRF_K_VALUE = 60
    VECTOR_SEARCH_CANDIDATES = 800
    TEXT_SEARCH_LIMIT = 20
```

### 에러 핸들링
```python
class RAGException(Exception):
    """RAG 서비스 기본 예외"""
    pass

class SearchException(RAGException):
    """검색 관련 예외"""
    pass

class GenerationException(RAGException):
    """답변 생성 관련 예외"""
    pass

# 사용 예시
try:
    search_results = await self.hybrid_search(query)
    if not search_results:
        raise SearchException("No relevant documents found")
    
    answer = await self.generate_answer(query, search_results)
    if not answer:
        raise GenerationException("Failed to generate answer")
        
except SearchException as e:
    logger.warning(f"Search failed: {e}")
    return {"success": False, "error": "search_failed"}
```

## 📝 커밋 메시지 규칙

### RAG 관련 커밋 타입
- `feat(search)`: 검색 기능 개선
- `feat(rag)`: RAG 파이프라인 기능 추가
- `fix(retrieval)`: 문서 검색 버그 수정
- `perf(embedding)`: 임베딩 성능 최적화
- `docs(api)`: API 문서 업데이트

### 예시
```
feat(search): implement adaptive RRF fusion algorithm

- Add dynamic weight adjustment based on query type
- Improve retrieval accuracy by 15% on benchmark dataset
- Add comprehensive unit tests for fusion logic
- Update performance benchmarks

Fixes #87
Closes #91
```

## 🏗 Pull Request 가이드라인

### RAG PR 체크리스트
- [ ] 검색 정확도 테스트 통과
- [ ] 답변 품질 평가 완료
- [ ] 성능 벤치마크 확인
- [ ] 메모리 사용량 체크
- [ ] 다양한 질문 유형 테스트

### 검색 품질 평가
```python
# 검색 정확도 측정
def evaluate_search_quality(test_queries: List[str]) -> Dict[str, float]:
    """검색 품질 평가 메트릭"""
    metrics = {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0, 
        "mrr": 0.0,  # Mean Reciprocal Rank
        "ndcg": 0.0  # Normalized Discounted Cumulative Gain
    }
    
    for query in test_queries:
        # 검색 실행 및 평가
        results = search_function(query)
        # 메트릭 계산
        
    return metrics
```

### 답변 품질 평가
- **관련성**: 질문과 답변의 연관성 (1-5점)
- **정확성**: 답변 내용의 정확도 (1-5점)
- **완성도**: 답변의 완전성 (1-5점)
- **명확성**: 답변의 이해하기 쉬움 (1-5점)

## 🎯 성능 기준

### 검색 성능 목표
- **검색 시간**: 평균 500ms 이하
- **정확도**: Precision@5 80% 이상
- **재현율**: Recall@10 90% 이상
- **답변 품질**: 평균 4.0/5.0 이상

### 시스템 성능 목표
- **응답 시간**: 95th percentile 8초 이하
- **동시 처리**: 50개 요청
- **메모리 사용**: 요청당 100MB 이하
- **가용성**: 99.5% 이상

## 🔄 데이터 관리

### 문서 인덱싱
```python
# 새로운 문서 추가 워크플로우
async def add_documents(documents: List[Document]) -> bool:
    """문서 추가 및 인덱싱"""
    
    # 1. 문서 전처리
    processed_docs = [preprocess_document(doc) for doc in documents]
    
    # 2. 청킹 (문서 분할)
    chunks = []
    for doc in processed_docs:
        doc_chunks = chunk_document(doc, chunk_size=1000, overlap=200)
        chunks.extend(doc_chunks)
    
    # 3. 임베딩 생성
    embeddings = await generate_embeddings_batch(chunks)
    
    # 4. 데이터베이스 저장
    await self.collection.insert_many([
        {
            "content": chunk.content,
            "source": chunk.source,
            "page_number": chunk.page,
            "embedding": embedding,
            "metadata": chunk.metadata
        }
        for chunk, embedding in zip(chunks, embeddings)
    ])
    
    return True
```

### 인덱스 관리
```javascript
// MongoDB Atlas Vector Search 인덱스
{
  "type": "vectorSearch",
  "name": "vector_index",
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "metadata.category"
    }
  ]
}

// Atlas Text Search 인덱스
{
  "type": "search", 
  "name": "text_index",
  "mappings": {
    "dynamic": false,
    "fields": {
      "content": {
        "type": "string",
        "analyzer": "korean"
      }
    }
  }
}
```

## 📞 커뮤니케이션

### RAG 관련 이슈 라벨
- `search-accuracy`: 검색 정확도 관련
- `answer-quality`: 답변 품질 관련
- `performance`: 성능 최적화
- `embedding`: 벡터 임베딩 관련
- `database`: MongoDB 관련

### 이슈 템플릿
```markdown
## 🔍 RAG 품질 이슈
**질문**: "구체적인 질문 내용"
**기대 답변**: "예상되는 답변"
**실제 답변**: "시스템이 생성한 답변"
**검색된 문서**: "관련 문서 제목들"
**평가 점수**: 관련성/정확성/완성도/명확성 (각 1-5점)
**환경**: MongoDB 버전, Azure OpenAI 모델 등
```

## 🚀 배포 가이드

### 프로덕션 체크리스트
- [ ] 벡터 인덱스 최적화 완료
- [ ] 대용량 데이터셋 테스트 완료
- [ ] 부하 테스트 통과
- [ ] 보안 검사 완료
- [ ] 모니터링 설정 완료

### 성능 모니터링
```python
# 성능 메트릭 수집
@app.middleware("http")
async def performance_monitor(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # 메트릭 로깅
    logger.info({
        "endpoint": request.url.path,
        "method": request.method,
        "process_time": process_time,
        "status_code": response.status_code
    })
    
    return response
```

## 📄 라이선스

기여하신 코드는 [MIT License](LICENSE)에 따라 배포됩니다.

---

RAG Service를 더욱 지능적으로 만들어주셔서 감사합니다! 🎉🤖
