# Import the required modules
from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Create a flask app
app = Flask(__name__)

# Load the dialogpt model and tokenizer
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")

# Define a function to generate a response from the chatbot
def chatbot_response(user_input):
  # Encode the user input and add the end-of-string token
  user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
  # Generate a response from the model
  response_ids = model.generate(user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
  # Decode the response and remove the end-of-string token
  response = tokenizer.decode(response_ids[:, user_input_ids.shape[-1]:][0], skip_special_tokens=True)
  return response

# Define a route for the home page
@app.route("/")
def home():
  return render_template("home.html")

# Define a route for the chatbot response
@app.route("/get")
def get_bot_response():
  # Get the user input from the request args
  user_input = request.args.get("msg")
  # Generate a response from the chatbot
  response = chatbot_response(user_input)
  return response

# Run the app
if __name__ == "__main__":
  app.run()
