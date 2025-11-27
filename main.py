# Import the required libraries
import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image
import litserve as ls


class SigLIP2API(ls.LitAPI):
    """
    SigLIP2API is a subclass of ls.LitAPI that provides an interface to the SigLIP2 model for zero-shot classification.

    Methods:
        - setup(device): Called once at startup for the task-specific setup.
        - decode_request(request): Convert the request payload to model input.
        - predict(model_inputs): Uses the model to predict the classification results.
        - encode_response(output): Convert the model output to a response payload.
    """

    def setup(self, device):
        """
        Set up the model for zero-shot classification.
        """
        model_id = "google/siglip2-base-patch16-224"
        self.device = device
        self.model = AutoModel.from_pretrained(model_id).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def decode_request(self, request):
        """
        Convert the request payload to model input.
        """
        # Extract the image path and labels from the request
        image = load_image(request["image_path"])
        labels = request["labels"]

        # Prepare the model inputs
        inputs = self.processor(
            text=labels,
            images=[image],
            return_tensors="pt",
            max_num_patches=256,
        ).to(self.model.device)

        # Return the model inputs
        return inputs, labels

    def predict(self, model_inputs):
        """
        Run inference and get the model output.
        """
        inputs, labels = model_inputs

        # Run inference with the model
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = torch.sigmoid(logits_per_image)

        # Return the model output as a list of predictions
        return sorted(
            [
                {"label": label, "score": f"{round(p.item() * 100, 2):.2f}%"}
                for label, p in zip(labels, probs[0])
            ],
            key=lambda x: float(x["score"][:-1]),
            reverse=True,
        )

    def encode_response(self, output):
        """
        Convert the model output to a response payload.
        """
        return {"predictions": output}


if __name__ == "__main__":
    # Create an instance of the SigLIP2API class and run the server
    api = SigLIP2API()
    server = ls.LitServer(api, track_requests=True)
    server.run(port=8000)