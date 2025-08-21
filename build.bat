@echo off
echo 🤖 RAG Service Docker 이미지 빌드 중...

REM 기존 이미지 삭제 (선택사항)
docker rmi rag-service:latest 2>nul

REM 이미지 빌드
docker build -t rag-service:latest .

if %errorlevel% equ 0 (
    echo ✅ RAG Service 이미지 빌드 완료!
    echo 이미지 크기:
    docker images rag-service:latest
) else (
    echo ❌ RAG Service 이미지 빌드 실패!
    exit /b 1
)

pause
