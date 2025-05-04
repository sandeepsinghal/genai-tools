import pytest
import os
import sys
from unittest.mock import patch, Mock
import tempfile

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from token_counter import count_tokens_ollama, count_tokens_openai

@pytest.fixture
def sample_text_file():
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("Hello, this is a test document.")
    yield f.name
    os.unlink(f.name)

@pytest.fixture
def mock_tiktoken():
    with patch('tiktoken.encoding_for_model') as mock_encoding:
        mock_enc = Mock()
        mock_enc.encode.return_value = [100, 200, 300, 400]  # Simulate 4 tokens
        mock_encoding.return_value = mock_enc
        yield mock_encoding

@pytest.fixture
def mock_requests_post():
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"tokens": ["token1", "token2", "token3"]}
        mock_post.return_value = mock_response
        yield mock_post

def test_count_tokens_ollama(sample_text_file, mock_requests_post):
    # Test Ollama token counting
    token_count = count_tokens_ollama(sample_text_file)
    assert token_count == 3
    mock_requests_post.assert_called_once()
    
    # Test error handling
    mock_requests_post.return_value.status_code = 500
    mock_requests_post.return_value.text = "Server Error"
    with pytest.raises(Exception, match="Ollama API error: Server Error"):
        count_tokens_ollama(sample_text_file)

def test_count_tokens_openai(sample_text_file, mock_tiktoken):
    # Test OpenAI token counting
    token_count = count_tokens_openai(sample_text_file)
    assert token_count == 4
    mock_tiktoken.assert_called_once_with("gpt-3.5-turbo")
