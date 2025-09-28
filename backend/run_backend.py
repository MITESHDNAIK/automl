#!/usr/bin/env python3
"""
Startup script for AutoML FastAPI backend
Run this file to start the backend server with proper configuration
"""

import uvicorn
import os
import sys

def main():
    print("🚀 Starting AutoML Backend Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API docs will be available at: http://localhost:8000/docs")
    print("❤️ Health check: http://localhost:8000/health")
    print("-" * 50)
    
    # Ensure required directories exist
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    try:
        # Run the FastAPI server
        uvicorn.run(
            "main:app",
            host="0.0.0.0",  # Allow external connections
            port=8000,
            reload=True,     # Auto-reload on code changes
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()