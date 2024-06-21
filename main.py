import sys
from pathlib import Path

# Fix path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'llmchat2'))

from llmchat.config import Config
from llmchat.client import DiscordClient

def main():
    config = Config()
    client = DiscordClient(config)

if __name__ == "__main__":
    main()