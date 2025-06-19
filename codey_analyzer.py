from google.cloud import aiplatform

# Initialize the Vertex AI client
aiplatform.init(project='famous-augury-463217-s2', location='us-central1') # Replace with your project ID and location

# Example: Code completion (replace with your desired Codey API task)
model = aiplatform.CodeGenerationModel.from_pretrained("code-gecko@001")

response = model.predict(
    prefix="def hello_world():\n    ",
    temperature=0.2,
    max_output_tokens=256
)

print(response.predictions[0])
