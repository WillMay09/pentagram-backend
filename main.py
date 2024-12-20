import modal
import io
from fastapi import Response, HTTPException,Query,Request
from datetime import datetime, timezone
import requests
import os #access to env variables

#downloads pretrained model from huggingFace
def download_model():
    #libraries needed for converting text to image
    from diffusers import AutoPipelineForText2Image
    import torch

    AutoPipelineForText2Image.from_pretrained(

        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,#precision
        variant="fp16"#modals format


    )
    #configuring the docker image for modal
image = (modal.Image.debian_slim()
         .pip_install("fastapi[standard]", "transformers","accelerate","diffusers","requests")
         .run_function(download_model))

app = modal.App("sd-demo", image=image)

#registers a class-based application on Modal
@app.cls(image=image,
         gpu="A10G",
         container_idle_timeout=300,
         secrets=[modal.Secret.from_name("API_KEY")])

class Model:

    @modal.build()
    @modal.enter()
    #imports libraries again to ensure they are abailable in execution environment
    def load_weights(self):
        from diffusers import AutoPipelineForText2Image
        import torch

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant='fp16'

        )
        #moves the model to the GPU
        self.pipe.to("cuda")
        self.API_KEY = os.environ["API_KEY"]
    @modal.web_endpoint()
    def generate(self, request: Request, prompt: 
                 str=Query(..., description="The prompt for image generation")):
        
        api_key = request.headers.get("X-API-KEY")
        if api_key != self.API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized"

            )
        #runs model pipeline to generate image based on prompt
        image = self.pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        #saves image in memory buffer
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        return Response(content=buffer.getValue(), media_type="image/jpeg")
    @modal.web_endpoint()
    def health(self):
        """ Lightweight endpoint for keeping container warm"""
        return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
    #schedules the update_keep_warm function to run periodically
@app.function(schedule=modal.Cron("*/5 * * * *"))

def update_keep_warm():

    health_url = "https://willmay09--sd-demo-model-health.modal.run"
    generateImage_url = "https://willmay09--sd-demo-model-generate.modal.run"
    #make a request to the health endpoint
    health_response = requests.get(health_url)
    print(f"Health check at: {health_response.json()['timestamp']}")

    #then make a request to the generate endpoint with API key
    headers = {"X-API-KEY": os.environ["API_KEY"]}
    generate_response = requests.get(generateImage_url, headers=headers)
    print(f"Generate endpoint tested successfully at: {datetime.now(timezone.utc).
                                                       isoformat()}")

    