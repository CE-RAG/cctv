import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image
from PIL import Image
import litserve as ls
import numpy as np

class SigLIP2API(ls.LitAPI):
    """
    SigLIP2API supporting Zero-shot Classification, Image Embedding, and Text Embedding.
    """

    def setup(self, device):
        model_id = "google/siglip2-base-patch16-224"
        self.device = device
        self.model = AutoModel.from_pretrained(model_id).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        
    def get_image_embedding(self, image_path: str) -> np.ndarray:
        # load image
        try:
            image = load_image(image_path) # ใช้ load_image ของ transformers เพื่อความง่ายในการรับ URL/Path
            if not isinstance(image, Image.Image):
                 image = Image.fromarray(image).convert("RGB")
        except:
             image = Image.open(image_path).convert("RGB")
        
        # SigLIP docs: processor(images=image, return_tensors="pt")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
    
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)[0]  # (D,)
    
        # L2 normalize
        image_features = image_features / image_features.norm(p=2)
        return image_features.cpu().numpy()
        
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
        """
        image_path = request.get("image_path")
        labels = request.get("labels")
        text = request.get("text")

        # Case 1: Zero-shot Classification
        if image_path and labels:
            return {"task": "classification", "image_path": image_path, "labels": labels}
        
        # Case 2: Image Embedding
        elif image_path:
            return {"task": "image_embedding", "image_path": image_path}
            
        # Case 3: Text Embedding
        elif text:
            return {"task": "text_embedding", "text": text}
            
        else:
            raise ValueError("Invalid request format. Provide 'image_path' and 'labels', only 'image_path', or only 'text'.")

    def predict(self, inputs):
        task = inputs["task"]

        if task == "image_embedding":
            # Return Image Embedding Vector
            embedding = self.get_image_embedding(inputs["image_path"])
            return {"type": "image_embedding", "embedding": embedding.tolist()}

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
    server = ls.LitServer(api, track_requests=True)
    server.run(port=8000)