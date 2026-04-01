import os

# You can either set `OPENAI_API_KEY` environment variable or replace "YOUR_API_KEY" below with your OpenAI API key
if "OPENAI_API_KEY" in os.environ:
    api_key = os.environ["OPENAI_API_KEY"]
else:
    api_key = "YOUR_API_KEY"

# MiniMax API key: set `MINIMAX_API_KEY` environment variable or replace "YOUR_MINIMAX_API_KEY" below
if "MINIMAX_API_KEY" in os.environ:
    minimax_api_key = os.environ["MINIMAX_API_KEY"]
else:
    minimax_api_key = "YOUR_MINIMAX_API_KEY"
