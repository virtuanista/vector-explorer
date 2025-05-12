import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Config:
    """Gestiona la configuración del proyecto"""
    
    def __init__(self, config_file: str = None):
        if not config_file:
            config_file = Path(__file__).parent.parent.parent / 'config' / 'config.json'
            
        with open(config_file) as f:
            self.config = json.load(f)
    
    def get(self, key, default=None):
        """Obtiene un valor de la configuración"""
        if '.' in key:
            # Soporta acceso anidado como 'tensorboard.log_dir'
            keys = key.split('.')
            value = self.config
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k, default)
                else:
                    return default
            return value
        return self.config.get(key, default)
    
    def get_output_dir(self):
        """Obtiene el directorio de salida"""
        return self.get('output_dir', 'embeddings_output')
        
    def get_log_dir(self):
        """Obtiene el directorio de logs"""
        return self.get('tensorboard.log_dir', 'logs')
