from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import logging
from pathlib import Path
import os
import tqdm

from src.utils.config import Config

logger = logging.getLogger(__name__)

class ModelManager:
    """Gestor de modelos de Hugging Face"""
    
    AVAILABLE_MODELS = {
        "sentence-transformers/all-MiniLM-L6-v2": "MiniLM-L6 - Modelo ligero (80MB) optimizado para embeddings",
        "BAAI/bge-small-en-v1.5": "BGE-small - Modelo pequeño (120MB) con gran rendimiento",
        "intfloat/multilingual-e5-small": "E5-small - Modelo multilingüe pequeño (140MB)",
        "thenlper/gte-small": "GTE-small - General Text Embeddings pequeño (170MB)"
    }
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.model = None
        self.tokenizer = None
        self.downloads_dir = Path(__file__).parent / "downloads"
        self.downloads_dir.mkdir(exist_ok=True)
        
    def load_model(self, model_name: str = None) -> bool:
        """Carga el modelo y el tokenizer"""
        try:
            model_name = model_name or self.config.get_default_model()
            
            logger.info(f"Descargando modelo {model_name}...")
            
            cache_dir = str(self.downloads_dir / model_name.split('/')[-1])
            
            logger.info("Inicializando tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            logger.info("Cargando modelo...")
            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float32
            )
            
            logger.info("✅ Modelo cargado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error al cargar el modelo: {str(e)}")
            return False

    def get_embedding(self, text: str) -> torch.Tensor:
        """Obtiene el embedding de un texto usando el modelo cargado"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("No hay ningún modelo cargado")
            
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Usamos mean pooling sobre la última capa oculta
            embeddings = torch.mean(outputs.last_hidden_state, dim=1)
            
        return embeddings.squeeze()

    def extract_vocabulary_embeddings(self) -> tuple[torch.Tensor, list[str]]:
        """Extrae embeddings para todo el vocabulario del modelo"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("No hay ningún modelo cargado")
        
        logger.info("Extrayendo embeddings del vocabulario completo...")
        
        # Obtener todo el vocabulario
        vocab = self.tokenizer.get_vocab()
        vocab_tokens = sorted(vocab.items(), key=lambda x: x[1])  # Ordenar por ID
        vocab_words = [word for word, _ in vocab_tokens]
        
        # Procesar en lotes para eficiencia
        batch_size = 64
        all_embeddings = []
        
        for i in tqdm.tqdm(range(0, len(vocab_words), batch_size)):
            batch_tokens = vocab_words[i:i + batch_size]
            inputs = self.tokenizer(
                batch_tokens,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Usamos mean pooling sobre la última capa oculta
                embeddings = torch.mean(outputs.last_hidden_state, dim=1)
                all_embeddings.append(embeddings)
        
        # Concatenar todos los embeddings
        embeddings_matrix = torch.cat(all_embeddings, dim=0)
        
        return embeddings_matrix, vocab_words

    def list_available_models(self):
        """Lista los modelos disponibles con sus descripciones"""
        return self.AVAILABLE_MODELS
