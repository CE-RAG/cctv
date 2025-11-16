#!/usr/bin/env python3
"""
Import car embeddings to Qdrant vector database.
"""

import json
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams


def create_collection(client: QdrantClient, collection_name: str, vector_size: int = 384):
    """Create Qdrant collection with semantic and serialization vectors."""
    try:
        # Check if collection exists
        collections = client.get_collections()
        existing_collections = [col.name for col in collections.collections]
        
        if collection_name in existing_collections:
            print(f"Collection '{collection_name}' already exists.")
            response = input("Do you want to delete and recreate it? (y/N): ")
            if response.lower() == 'y':
                client.delete_collection(collection_name)
                print(f"Deleted existing collection '{collection_name}'")
            else:
                print("Using existing collection.")
                return
        
        # Create collection with multiple named vectors
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "semantic": VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                ),
                "serialization": VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            }
        )
        print(f"✓ Created collection '{collection_name}' with vector size {vector_size}")
        
    except Exception as e:
        print(f"Error creating collection: {e}")
        raise


def import_jsonl_to_qdrant(jsonl_file: str, client: QdrantClient, collection_name: str, batch_size: int = 100):
    """Import points from JSONL file to Qdrant."""
    if not os.path.exists(jsonl_file):
        raise FileNotFoundError(f"File not found: {jsonl_file}")
    
    points = []
    total_points = 0
    
    print(f"Reading points from {jsonl_file}...")
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                point = json.loads(line.strip())
                points.append(point)
                total_points += 1
                
                # Upload in batches
                if len(points) >= batch_size:
                    client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                    print(f"  Uploaded batch: {len(points)} points (total: {total_points})")
                    points = []
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    # Upload remaining points
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"  Uploaded final batch: {len(points)} points (total: {total_points})")
    
    print(f"✓ Successfully imported {total_points} points to collection '{collection_name}'")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Import car embeddings to Qdrant')
    parser.add_argument(
        '--host',
        default='localhost',
        help='Qdrant host (default: localhost)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=6333,
        help='Qdrant port (default: 6333)'
    )
    parser.add_argument(
        '--collection',
        default='car_embeddings',
        help='Collection name (default: car_embeddings)'
    )
    parser.add_argument(
        '--file',
        default='car_embeddings_qdrant.jsonl',
        help='JSONL file to import (default: car_embeddings_qdrant.jsonl)'
    )
    parser.add_argument(
        '--vector-size',
        type=int,
        default=384,
        help='Vector dimension (default: 384 for BGE-small)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for upload (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jsonl_file = os.path.join(script_dir, args.file)
    
    print("=" * 60)
    print("Qdrant Import Script")
    print("=" * 60)
    print(f"Host: {args.host}:{args.port}")
    print(f"Collection: {args.collection}")
    print(f"File: {jsonl_file}")
    print(f"Vector Size: {args.vector_size}")
    print()
    
    # Connect to Qdrant
    try:
        print(f"Connecting to Qdrant at {args.host}:{args.port}...")
        client = QdrantClient(host=args.host, port=args.port)
        
        # Test connection
        collections = client.get_collections()
        print("✓ Connected to Qdrant")
        print()
        
    except Exception as e:
        print(f"✗ Failed to connect to Qdrant: {e}")
        print("\nMake sure Qdrant is running:")
        print("  docker-compose -f docker-compose-qdrant.yml up -d")
        return
    
    # Create collection
    try:
        create_collection(client, args.collection, args.vector_size)
        print()
    except Exception as e:
        print(f"✗ Failed to create collection: {e}")
        return
    
    # Import data
    try:
        import_jsonl_to_qdrant(jsonl_file, client, args.collection, args.batch_size)
        print()
        print("=" * 60)
        print("✓ Import complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return


if __name__ == '__main__':
    main()

