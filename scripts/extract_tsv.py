#!/usr/bin/env python
"""
Script para extraer solo los archivos TSV sin visualización
"""
import argparse
import logging
from pathlib import Path
import numpy as np

from src.models.model_manager import ModelManager
from src.models.embedding_writer import EmbeddingWriter
from src.utils.config import Config

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Extraer embeddings de texto a archivos TSV sin visualización'
    )
    
    parser.add_argument('--text', type=str, help='Texto para generar embeddings')
    parser.add_argument('--input-file', type=str, help='Archivo de texto de entrada (una oración por línea)')
    parser.add_argument('--model', type=str, help='Nombre del modelo a usar')
    parser.add_argument('--output-dir', type=str, help='Directorio de salida para los archivos TSV')
    parser.add_argument('--output-prefix', type=str, help='Prefijo para los archivos de salida')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamaño del batch para procesar textos')
    
    return parser.parse_args()

def process_text(text: str, model_manager: ModelManager, writer: EmbeddingWriter, prefix: str = None):
    """Procesa un texto y guarda sus embeddings"""
    try:
        # Generar embeddings
        embeddings = model_manager.get_embeddings(text)
        
        # Guardar archivos
        tensor_file, metadata_file, _ = writer.save_embeddings(
            embeddings.numpy(),
            text.split(),  # Tokenización simple para metadatos
            prefix=prefix
        )
        
        logger.info(f"✅ Embeddings guardados en:")
        logger.info(f"   - Tensores: {tensor_file}")
        logger.info(f"   - Metadatos: {metadata_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error al procesar texto: {str(e)}")
        return False

def process_file(file_path: str, model_manager: ModelManager, writer: EmbeddingWriter,
                batch_size: int = 32, prefix: str = None):
    """Procesa un archivo de texto línea por línea"""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"El archivo {file_path} no existe")
            return False
            
        # Leer líneas del archivo
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
            
        if not texts:
            logger.warning("El archivo está vacío")
            return False
            
        total_lines = len(texts)
        logger.info(f"Procesando {total_lines} líneas...")
        
        # Procesar por lotes
        embeddings_list = []
        tokens_list = []
        
        for i in range(0, total_lines, batch_size):
            batch = texts[i:i + batch_size]
            batch_size_actual = len(batch)
            
            # Procesar cada texto en el lote
            for text in batch:
                embeddings = model_manager.get_embeddings(text)
                embeddings_list.append(embeddings.numpy())
                tokens_list.extend(text.split())
                
            logger.info(f"Procesados {min(i + batch_size_actual, total_lines)}/{total_lines} textos")
        
        # Guardar resultados combinados
        combined_embeddings = np.vstack(embeddings_list)
        
        prefix = prefix or file_path.stem
        tensor_file, metadata_file, _ = writer.save_embeddings(
            combined_embeddings,
            tokens_list,
            prefix=prefix
        )
        
        logger.info(f"✅ Embeddings guardados en:")
        logger.info(f"   - Tensores: {tensor_file}")
        logger.info(f"   - Metadatos: {metadata_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error al procesar archivo: {str(e)}")
        return False

def main():
    args = parse_args()
    
    if not args.text and not args.input_file:
        logger.error("Debes proporcionar un texto o un archivo de entrada")
        return 1
        
    # Inicializar componentes
    config = Config()
    
    # Si se especificó un directorio de salida, actualizarlo en la configuración
    if args.output_dir:
        config.set_output_dir(args.output_dir)
    
    model_manager = ModelManager(config)
    writer = EmbeddingWriter(config)
    
    try:
        # Cargar modelo
        model_manager.load_model(args.model)
        
        success = False
        if args.text:
            success = process_text(args.text, model_manager, writer, args.output_prefix)
        else:
            success = process_file(args.input_file, model_manager, writer, 
                                 args.batch_size, args.output_prefix)
            
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}")
        return 1

if __name__ == '__main__':
    exit(main())
