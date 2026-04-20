#!/bin/bash
# RAG System — backend + frontend durum kontrolü
echo "🔍 Service Status"

# Backend health endpoint'i hızlı cevap verir
if curl -sf --max-time 2 http://localhost:8000/health >/dev/null 2>&1; then
    echo "✅ Backend  :8000  UP"
else
    echo "❌ Backend  :8000  DOWN"
fi

# Streamlit response'ı yavaş olabilir — port listening kontrolü daha güvenilir
streamlit_port=""
for port in 8501 8502; do
    if lsof -i :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        streamlit_port=$port
        break
    fi
done

if [ -n "$streamlit_port" ]; then
    echo "✅ Streamlit :$streamlit_port  UP"
else
    echo "❌ Streamlit        DOWN"
fi
