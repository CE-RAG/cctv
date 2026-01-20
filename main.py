import litserve as ls
import numpy as np
import torch

from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image


class SigLIP2API(ls.LitAPI):
    """
    SigLIP2API supporting Zero-shot Classification, Image Embedding, and Text Embedding, and batch Image Embedding.
    """

    def setup(self, device="auto"):
        base_model_id = "google/siglip2-base-patch16-224"
        
        adapter_path = "path/to/your/finetuned_folder"

        # Automatically detect and use CUDA if available
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")
        
        base_model = AutoModel.from_pretrained(base_model_id)
        # 2. Load and attach the Adapter (LoRA)
        # This looks for adapter_config.json and adapter_model.safetensors
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # 3. Move to device and set to eval
        self.model.to(self.device).eval()

        # 4. Load Processor from your local path (to use your custom tokenizer/special tokens)
        self.processor = AutoProcessor.from_pretrained(adapter_path)

        # If using CUDA, optimize for GPU
        if self.device == "cuda":
            self.model = torch.compile(self.model)  # Optimize for GPU

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        # load image
        try:
            image = load_image(
                image_path
            )  # ใช้ load_image ของ transformers เพื่อความง่ายในการรับ URL/Path
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {e}")
            image = Image.open(image_path).convert("RGB")

        # SigLIP docs: processor(images=image, return_tensors="pt")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)[0]  # (D,)

        # L2 normalize
        image_features = image_features / image_features.norm(p=2)
        return image_features.cpu().numpy()

    def get_batch_image_embeddings(self, image_paths: list) -> list:
        """Process multiple images in a batch for efficiency."""
        batch_size = len(image_paths)
        images = []
        valid_paths = []

        # Load and validate all images first
        for i, image_path in enumerate(image_paths):
            try:
                image = load_image(image_path)
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(image).convert("RGB")
                images.append(image)
                valid_paths.append(image_path)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Append None to maintain batch indexing
                images.append(None)
                valid_paths.append(image_path)

        # Process images in smaller sub-batches to avoid memory issues
        sub_batch_size = 16  # Process 16 images at a time
        all_embeddings = []

        for i in range(0, batch_size, sub_batch_size):
            end_idx = min(i + sub_batch_size, batch_size)
            sub_batch_images = []

            # Collect valid images for this sub-batch
            for j in range(i, end_idx):
                if images[j] is not None:
                    sub_batch_images.append(images[j])

            if not sub_batch_images:
                # No valid images in this sub-batch
                for j in range(i, end_idx):
                    all_embeddings.append(None)
                continue

            # Process the sub-batch
            inputs = self.processor(images=sub_batch_images, return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                batch_features = self.model.get_image_features(**inputs)
                # L2 normalize
                batch_features = batch_features / batch_features.norm(
                    p=2, dim=1, keepdim=True
                )
                batch_features = batch_features.cpu().numpy()

            # Add embeddings to the results, maintaining original batch order
            sub_idx = 0
            for j in range(i, end_idx):
                if images[j] is not None:
                    all_embeddings.append(batch_features[sub_idx])
                    sub_idx += 1
                else:
                    all_embeddings.append(None)

        # Convert to list format and include path info
        results = []
        for path, embedding in zip(valid_paths, all_embeddings):
            if embedding is not None:
                results.append({"path": path, "embedding": embedding.tolist()})
            else:
                results.append({"path": path, "error": "Failed to process image"})
                
        return results

    def get_text_embedding(self, text: str) -> np.ndarray:
        inputs = self.processor(
            text=[text],
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)[0]  # (D,)

        text_features = text_features / text_features.norm(p=2)
        return text_features.cpu().numpy()

    def decode_request(self, request):
        """
        Determine the task based on input keys.
        Possible inputs:
        1. {"image_path": "...", "labels": [...]} -> Classification
        2. {"image_path": "..."} -> Image Embedding
        3. {"text": "..."} -> Text Embedding
        4. {"image_paths": [...]} -> Batch Image Embedding
        """
        image_path = request.get("image_path")
        image_paths = request.get("image_paths")
        labels = request.get("labels")
        text = request.get("text")

        # Case 1: Zero-shot Classification
        if image_path and labels:
            return {
                "task": "classification",
                "image_path": image_path,
                "labels": labels,
            }

        # Case 2: Image Embedding
        elif image_path and not image_paths:
            return {"task": "image_embedding", "image_path": image_path}

        # Case 3: Batch Image Embedding
        elif image_paths and isinstance(image_paths, list):
            return {"task": "batch_image_embedding", "image_paths": image_paths}

        # Case 4: Text Embedding
        elif text:
            return {"task": "text_embedding", "text": text}

        else:
            raise ValueError(
                "Invalid request format. Provide 'image_path' and 'labels', only 'image_path', 'image_paths' as a list, or only 'text'."
            )

    def predict(self, inputs):
        task = inputs["task"]

        if task == "image_embedding":
            # Return Image Embedding Vector
            embedding = self.get_image_embedding(inputs["image_path"])
            return {"type": "image_embedding", "embedding": embedding.tolist()}

        elif task == "batch_image_embedding":
            # Process multiple images efficiently
            results = self.get_batch_image_embeddings(inputs["image_paths"])
            return {"type": "batch_image_embedding", "results": results}

        elif task == "text_embedding":
            # Return Text Embedding Vector
            embedding = self.get_text_embedding(inputs["text"])
            return {"type": "text_embedding", "embedding": embedding.tolist()}

        elif task == "classification":
            # Existing Classification Logic
            image_path = inputs["image_path"]
            labels = inputs["labels"]

            image = load_image(image_path)

            # Prepare inputs for classification
            model_inputs = self.processor(
                text=labels,
                images=[image],
                return_tensors="pt",
                max_num_patches=256,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**model_inputs)
                logits_per_image = outputs.logits_per_image
                probs = torch.sigmoid(logits_per_image)

            results = sorted(
                [
                    {"label": label, "score": f"{round(p.item() * 100, 2):.2f}%"}
                    for label, p in zip(labels, probs[0])
                ],
                key=lambda x: float(x["score"][:-1]),
                reverse=True,
            )
            return {"type": "classification", "results": results}

    def encode_response(self, output):
        return output


if __name__ == "__main__":
    api = SigLIP2API()
    # Setup with GPU acceleration
    api.setup(device="auto")  # Will automatically use CUDA if available
    server = ls.LitServer(api, track_requests=True)
    server.run(port=8000)
