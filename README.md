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
## Zero-shot classification
```sh
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "labels": ["I love u 2000"]
    "image_path": "https://example.com/image.jpg"
  }'
```
