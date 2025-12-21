# CCTV Siglip

## Text embeddings
```sh
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "liver pool 1-4 psv so noob"
  }'
```
## Image embeddings
```sh
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "https://example.com/image.jpg"
  }'
```
## Batch Image embeddings
```sh
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_paths": ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
  }'
```

### Response Structure
The batch image embedding endpoint returns a JSON object with the following structure:

```json
{
  "type": "batch_image_embedding",
  "results": [
    {
      "path": "path/to/image1.jpg",
      "embedding": [0.123, 0.456, 0.789, ...] // 768-dimensional embedding vector
    },
    {
      "path": "path/to/image2.jpg",
      "embedding": [0.987, 0.654, 0.321, ...] // 768-dimensional embedding vector
    },
    {
      "path": "path/to/image3.jpg",
      "error": "Failed to process image" // Error message if image couldn't be processed
    }
    // ... more results
  ]
}
```
## Zero-shot classification
```sh
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "labels": ["I love u 2000"]
    "image_path": "https://example.com/image.jpg"
  }'
```
