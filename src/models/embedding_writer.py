import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import os
import torch

from src.utils.config import Config

logger = logging.getLogger(__name__)

class EmbeddingWriter:
    """Gestiona la escritura de embeddings y metadatos"""
    
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(config.get_output_dir())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_embeddings(self, embeddings: np.ndarray, tokens: list, prefix: str = None) -> tuple:
        """
        Guarda los embeddings y tokens en archivos TSV
        
        Args:
            embeddings: Array de embeddings
            tokens: Lista de tokens correspondientes
            prefix: Prefijo opcional para los archivos
            
        Returns:
            tuple: (tensor_file, metadata_file, config_file)
        """
        # Generar timestamp para archivos únicos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_prefix = f"{prefix}_{timestamp}" if prefix else timestamp
        
        # Preparar archivos
        tensor_file = self.output_dir / f"{file_prefix}_tensor.tsv"
        metadata_file = self.output_dir / f"{file_prefix}_metadata.tsv"
        config_file = self.output_dir / f"{file_prefix}_projector_config.pbtxt"
        
        # Guardar embeddings
        logger.info(f"Guardando embeddings en {tensor_file}...")
        np.savetxt(tensor_file, embeddings, delimiter='\t')
        
        # Guardar tokens
        logger.info(f"Guardando tokens en {metadata_file}...")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write("token\n")  # Encabezado
            for token in tokens:
                f.write(f"{token}\n")
        
        # Crear configuración para TensorBoard
        logger.info(f"Creando configuración en {config_file}...")
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write('embeddings {\n')
            f.write('  tensor_name: "embeddings"\n')
            f.write(f'  tensor_path: "{tensor_file.name}"\n')
            f.write(f'  metadata_path: "{metadata_file.name}"\n')
            f.write('}\n')
            
        return tensor_file, metadata_file, config_file
        
    def get_latest_embeddings(self) -> tuple:
        """
        Obtiene los archivos de embeddings más recientes
        
        Returns:
            tuple: (tensor_file, metadata_file, config_file) o (None, None, None)
        """
        try:
            # Buscar archivos de configuración
            config_files = list(self.output_dir.glob("*_projector_config.pbtxt"))
            if not config_files:
                return None, None, None
                
            # Ordenar por fecha de modificación
            latest_config = max(config_files, key=lambda f: f.stat().st_mtime)
            base_name = latest_config.stem.rsplit('_projector_config', maxsplit=1)[0]
            
            tensor_file = self.output_dir / f"{base_name}_tensor.tsv"
            metadata_file = self.output_dir / f"{base_name}_metadata.tsv"
            
            if tensor_file.exists() and metadata_file.exists():
                return tensor_file, metadata_file, latest_config
                
        except Exception as e:
            logger.error(f"Error al buscar embeddings recientes: {str(e)}")
            
        return None, None, None
        
    def list_available_embeddings(self) -> list:
        """
        Lista todos los conjuntos de embeddings disponibles
        
        Returns:
            list: Lista de tuplas (nombre_base, tensor_file, metadata_file, config_file)
        """
        try:
            results = []
            config_files = list(self.output_dir.glob("*_projector_config.pbtxt"))
            
            for config_file in config_files:
                base_name = config_file.stem.rsplit('_projector_config', maxsplit=1)[0]
                tensor_file = self.output_dir / f"{base_name}_tensor.tsv"
                metadata_file = self.output_dir / f"{base_name}_metadata.tsv"
                
                if tensor_file.exists() and metadata_file.exists():
                    # Si el nombre tiene timestamp, lo extraemos para mostrar solo el modelo
                    display_name = base_name.split('_')[0] if '_' in base_name else base_name
                    results.append((
                        display_name,
                        tensor_file,
                        metadata_file,
                        config_file
                    ))
            
            return sorted(results, key=lambda x: x[1].stat().st_mtime, reverse=True)
            
        except Exception as e:
            logger.error(f"Error al listar embeddings: {str(e)}")
            return []
        
    def save_batch_embeddings(self, embeddings_list: list, texts: list, model_name: str):
        """Guarda un conjunto de embeddings y sus metadatos para TensorBoard"""
        try:
            if isinstance(embeddings_list[0], torch.Tensor):
                embeddings_array = torch.stack(embeddings_list).detach().cpu().numpy()
            else:
                embeddings_array = np.stack(embeddings_list)
                
            # Guardar embeddings
            np.savetxt(
                self.output_dir / "tensor.tsv",
                embeddings_array,
                delimiter='\t'
            )
            
            # Guardar metadatos
            with open(self.output_dir / "metadata.tsv", 'w', encoding='utf-8') as f:
                f.write('\n'.join(texts))
                
        except Exception as e:
            logger.error(f"Error al guardar embeddings por lote: {str(e)}")
