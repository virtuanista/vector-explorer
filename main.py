#!/usr/bin/env python
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from src.cli.interactive import CLI

if __name__ == "__main__":
    cli = CLI()
    cli.run()
