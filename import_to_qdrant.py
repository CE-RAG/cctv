#!/usr/bin/env python3
"""
Import car embeddings to Qdrant vector database.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from transformers import AutoModel, AutoProcessor


def create_collection(
    client: QdrantClient, collection_name: str, vector_size: int = 768
):
    """Create Qdrant collection with semantic and serialization vectors."""
    try:
        # Check if collection exists
        collections = client.get_collections()
        existing_collections = [col.name for col in collections.collections]

        if collection_name in existing_collections:
            print(f"Collection '{collection_name}' already exists.")
            response = input("Do you want to delete and recreate it? (y/N): ")
            if response.lower() == "y":
                client.delete_collection(collection_name)
                print(f"Deleted existing collection '{collection_name}'")
            else:
                print("Using existing collection.")
                return

        # Create collection with SigLIP2 embeddings
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(
            f"✓ Created collection '{collection_name}' with vector size {vector_size}"
        )

    except Exception as e:
        print(f"Error creating collection: {e}")
        raise


def import_jsonl_to_qdrant(
    jsonl_file: str, client: QdrantClient, collection_name: str, batch_size: int = 100
):
    """Import points from JSONL file to Qdrant."""
    if not os.path.exists(jsonl_file):
        raise FileNotFoundError(f"File not found: {jsonl_file}")

    points = []
    total_points = 0

    print(f"Reading points from {jsonl_file}...")
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                point = json.loads(line.strip())
                points.append(point)
                total_points += 1

                # Upload in batches
                if len(points) >= batch_size:
                    client.upsert(collection_name=collection_name, points=points)
                    print(
                        f"  Uploaded batch: {len(points)} points (total: {total_points})"
                    )
                    points = []

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

    # Upload remaining points
    if points:
        client.upsert(collection_name=collection_name, points=points)
        print(f"  Uploaded final batch: {len(points)} points (total: {total_points})")

    print(
        f"✓ Successfully imported {total_points} points to collection '{collection_name}'"
    )


def generate_siglip2_embeddings(
    image_dir: str,
    output_file: str,
    model_name: str = "google/siglip2-base-patch16-224",
):
    """Generate SigLIP2 embeddings for all images in a directory.

    Args:
        image_dir: Directory containing images
        output_file: Output JSONL file path
        model_name: SigLIP2 model name
    """
    import torch
    from transformers.utils import is_flash_attn_2_available

    print(f"Loading SigLIP2 model: {model_name}")

    # Use flash attention if available
    attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None

    # Load model and processor
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model loaded on {device}")

    # Process all images
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(Path(image_dir).glob(ext))

    print(f"Found {len(image_files)} images")

    with open(output_file, "w") as f:
        for i, image_path in enumerate(image_files):
            try:
                # Load and process image
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Generate embedding
                with torch.no_grad():
                    image_embeds = model.get_image_features(**inputs)
                    # Normalize embeddings
                    image_embeds = image_embeds / image_embeds.norm(
                        p=2, dim=-1, keepdim=True
                    )

                # Convert to list for JSON serialization
                embedding = image_embeds.cpu().numpy().flatten().tolist()

                # Create point data
                point_data = {
                    "id": str(image_path.stem),
                    "vector": embedding,
                    "payload": {
                        "image_path": str(image_path),
                        "filename": image_path.name,
                        "created_at": datetime.now().isoformat(),
                    },
                }

                # Write to JSONL
                f.write(json.dumps(point_data) + "\n")

                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(image_files)} images")

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    print(f"✓ Generated embeddings for {len(image_files)} images")
    print(f"✓ Saved to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Import car embeddings to Qdrant")
    parser.add_argument(
        "--host", default="localhost", help="Qdrant host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=6333, help="Qdrant port (default: 6333)"
    )
    parser.add_argument(
        "--collection",
        default="car_embeddings",
        help="Collection name (default: car_embeddings)",
    )
    parser.add_argument(
        "--file",
        default="car_qdrant.jsonl",
        help="JSONL file to import (default: car_qdrant.jsonl)",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=768,
        help="Vector dimension (default: 768 for SigLIP2-base)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for upload (default: 100)",
    )
    parser.add_argument(
        "--generate-embeddings",
        action="store_true",
        help="Generate embeddings from images before importing",
    )
    parser.add_argument(
        "--image-dir",
        default="car_images",
        help="Directory containing images (default: car_images)",
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

    # Generate embeddings if requested
    if args.generate_embeddings:
        image_dir = os.path.join(script_dir, args.image_dir)
        if not os.path.exists(image_dir):
            print(f"✗ Image directory not found: {image_dir}")
            return

        print("Generating SigLIP2 embeddings...")
        generate_siglip2_embeddings(image_dir, jsonl_file)
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
        print("  docker-compose up -d")
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


if __name__ == "__main__":
    main()
