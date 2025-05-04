import argparse
import os
import requests

try:
    import openai
except ImportError:
    openai = None

def count_tokens_openai(file_path, model="gpt-3.5-turbo"):
    if openai is None:
        raise ImportError("openai package not installed. Run `pip install openai`.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Use tiktoken for token counting if available
    try:
        import tiktoken
    except ImportError:
        raise ImportError("tiktoken package not installed. Run `pip install tiktoken` for accurate counting.")
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(content)
    return len(tokens)

def count_tokens_ollama(file_path, model="llama3"):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Ollama's /api/tokenize endpoint
    url = "http://localhost:11434/api/tokenize"
    data = {"model": model, "prompt": content}
    resp = requests.post(url, json=data)
    if resp.status_code != 200:
        raise Exception(f"Ollama API error: {resp.text}")
    tokens = resp.json().get("tokens", [])
    return len(tokens)

def main():
    parser = argparse.ArgumentParser(description="Count tokens in a file using OpenAI or Ollama.")
    parser.add_argument("file", help="Path to the input file.")
    parser.add_argument("--provider", choices=["openai", "ollama"], required=True, help="Token counter provider.")
    parser.add_argument("--model", default=None, help="Model name (default: gpt-3.5-turbo for OpenAI, llama3 for Ollama)")
    args = parser.parse_args()

    if args.provider == "openai":
        model = args.model or "gpt-3.5-turbo"
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Please set the OPENAI_API_KEY environment variable.")
            exit(1)
        openai.api_key = api_key
        n_tokens = count_tokens_openai(args.file, model)
        print(f"Number of tokens (OpenAI, model={model}): {n_tokens}")
    else:
        model = args.model or "llama3"
        n_tokens = count_tokens_ollama(args.file, model)
        print(f"Number of tokens (Ollama, model={model}): {n_tokens}")

if __name__ == "__main__":
    main()