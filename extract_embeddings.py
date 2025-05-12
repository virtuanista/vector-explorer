from src.cli.interactive import start_cli
from src.utils.config import Config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

def main():
    config = Config()
    start_cli(config)

if __name__ == "__main__":
    main()
