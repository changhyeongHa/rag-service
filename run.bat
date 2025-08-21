@echo off
echo 🤖 RAG Service 실행 중...

REM 기존 컨테이너 정리
docker stop rag-service 2>nul
docker rm rag-service 2>nul

REM 환경변수 파일 경로 (상위 디렉토리)
set ENV_FILE=..\.env

REM 컨테이너 실행
docker run -d ^
  --name rag-service ^
  -p 8002:8002 ^
  --env-file %ENV_FILE% ^
  --restart unless-stopped ^
  rag-service:latest

if %errorlevel% equ 0 (
    echo ✅ RAG Service 시작 완료!
    echo 포트: http://localhost:8002
    echo 헬스체크: http://localhost:8002/health
    echo API 문서: http://localhost:8002/docs
    echo.
    echo 로그 확인: docker logs -f rag-service
    echo 컨테이너 정지: docker stop rag-service
) else (
    echo ❌ RAG Service 시작 실패!
    exit /b 1
)

pause
