#!/bin/bash
# Setup script for Org Knowledge Management Chatbot with ChromaDB

set -e

echo "🚀 Setting up Org Knowledge Management Chatbot with ChromaDB..."

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop first."
    echo "   Download from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose."
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+."
    exit 1
fi

echo "✅ Prerequisites check passed!"

# Load environment variables
echo "📋 Loading environment variables from .env file..."
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please create one based on the migration guide."
    exit 1
fi

set -a
source .env
set +a

# Validate required variables
echo "🔍 Validating required environment variables..."
required_vars=("OPENAI_API_KEY" "MONGO_DEV_PASSWORD" "JWT_SECRET" "JWT_REFRESH_SECRET" "CREDS_KEY" "CREDS_IV")

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Required variable $var is not set in .env file"
        exit 1
    fi
done

echo "✅ Environment configuration loaded successfully!"

# Create directories
echo "📁 Creating project directories..."
mkdir -p logs
echo "✅ Directories created!"

echo ""
echo "📁 Document Organization:"
echo "   Place your documents in:"
echo "   • data/documents/approved/  - For approved/successful projects"
echo "   • data/documents/rejected/  - For rejected/unsuccessful proposals"
echo ""

# Stop any existing services
echo "🛑 Stopping any existing services..."
docker-compose down -v 2>/dev/null || true

# Pull/build images
echo "🐳 Pulling/building Docker images..."
docker-compose pull
docker-compose build

# Start services
echo "🚀 Starting Docker services..."
docker-compose up -d

echo "⏳ Waiting for services to be ready..."
sleep 15

# Check ChromaDB health
echo "🏥 Checking ChromaDB health..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8000/api/v2/heartbeat > /dev/null 2>&1; then
        echo "✅ ChromaDB is healthy!"
        break
    fi
    echo "   Waiting for ChromaDB... (attempt $((attempt+1))/$max_attempts)"
    sleep 5
    attempt=$((attempt+1))
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ ChromaDB failed to start. Check logs with: docker-compose logs chromadb"
    exit 1
fi

# Check RAG backend health
echo "🏥 Checking RAG backend health..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "✅ RAG backend is healthy!"
        break
    fi
    echo "   Waiting for RAG backend... (attempt $((attempt+1))/$max_attempts)"
    sleep 5
    attempt=$((attempt+1))
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ RAG backend failed to start. Check logs with: docker-compose logs rag-backend"
    exit 1
fi

# Check if documents directory has files
doc_count=$(find data/documents -type f \( -name "*.docx" -o -name "*.xlsx" -o -name "*.pdf" \) 2>/dev/null | wc -l)
if [ "$doc_count" -eq 0 ]; then
    echo ""
    echo "📄 No documents found in data/documents/"
    echo "   Please add your Word, Excel, or PDF files to:"
    echo "   • data/documents/approved/   (for approved projects)"
    echo "   • data/documents/rejected/   (for rejected proposals)"
    echo ""
    echo "   Then run: python ingest_documents.py data/documents"
else
    echo ""
    echo "📄 Found $doc_count documents. Would you like to ingest them now? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "📥 Installing Python dependencies for ingestion..."
        pip3 install -r requirements.txt --quiet
        
        echo "📥 Starting document ingestion..."
        python3 ingest_documents.py data/documents --api-url http://localhost:8001
        
        echo ""
        echo "✅ Document ingestion completed!"
    else
        echo "   You can ingest documents later with:"
        echo "   python ingest_documents.py data/documents --api-url http://localhost:8001"
    fi
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "🌐 Access your services:"
echo "   • LibreChat UI:     http://localhost:3080"
echo "   • RAG API:          http://localhost:8001"
echo "   • API Docs:         http://localhost:8001/docs"
echo "   • ChromaDB:         http://localhost:8000"
echo "   • Stats:            http://localhost:8001/stats"
echo ""
echo "📊 View statistics:"
echo "   curl http://localhost:8001/stats | jq"
echo ""
echo "🔧 Useful commands:"
echo "   • View logs:        docker-compose logs -f"
echo "   • Stop services:    docker-compose down"
echo "   • Restart:          docker-compose restart"
echo "   • Ingest docs:      python ingest_documents.py data/documents --api-url http://localhost:8001"
echo ""
echo "📖 First time setup:"
echo "   1. Go to http://localhost:3080"
echo "   2. Create your admin account"
echo "   3. Select 'Org Knowledge Base' as the model"
echo "   4. Start chatting with your knowledge base!"
echo ""
echo "💡 Try these example queries:"
echo "   • 'Show me approved projects'"
echo "   • 'What rejected proposals do we have from 2023?'"
echo "   • 'Tell me about successful livestock projects'"
echo ""
echo "📁 Document Organization Reminder:"
echo "   • Approved projects → data/documents/approved/"
echo "   • Rejected proposals → data/documents/rejected/"
echo ""