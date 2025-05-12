import logging
import inquirer
from src.models.model_manager import ModelManager
from src.models.embedding_writer import EmbeddingWriter
from src.utils.tensorboard import TensorBoardManager
from src.utils.config import Config

logger = logging.getLogger(__name__)

class CLI:
    """Interfaz de línea de comandos interactiva"""
    
    def __init__(self):
        self.config = Config()
        self.model_manager = ModelManager(self.config)
        self.embedding_writer = EmbeddingWriter(self.config)
        self.tensorboard = TensorBoardManager(self.config)
    
    def run(self):
        """Ejecuta la interfaz interactiva"""
        logger.info("\n¡Bienvenido al explorador de embeddings!")
        logger.info("Este programa te permite extraer y visualizar embeddings de modelos.")
        
        while True:
            questions = [
                inquirer.List('action',
                    message="¿Qué acción deseas realizar?",
                    choices=[
                        ('Extraer embeddings', 'extract'),
                        ('Visualizar embeddings existentes', 'visualize'),
                        ('Salir', 'exit')
                    ]
                )
            ]
            
            answers = inquirer.prompt(questions)
            
            if answers['action'] == 'extract':
                self._extract_embeddings()
            elif answers['action'] == 'visualize':
                self._visualize_embeddings()
            else:
                logger.info("¡Hasta luego!")
                break

    def _extract_embeddings(self):
        """Extrae embeddings de un modelo seleccionado por el usuario"""
        # Seleccionar modelo
        models = self.model_manager.list_available_models()
        questions = [
            inquirer.List('model',
                message="Selecciona el modelo a utilizar:",
                choices=[(desc, name) for name, desc in models.items()]
            )
        ]
        
        answers = inquirer.prompt(questions)
        if not answers:
            return
            
        model_name = answers['model']
        
        # Cargar modelo
        logger.info(f"\nCargando modelo {model_name}...")
        if not self.model_manager.load_model(model_name):
            return

        try:
            logger.info("\nExtrayendo embeddings del vocabulario completo...")
            embeddings_matrix, vocab_words = self.model_manager.extract_vocabulary_embeddings()
            
            self.embedding_writer.save_batch_embeddings(
                embeddings_matrix, 
                vocab_words, 
                model_name
            )
            logger.info("✅ Embeddings del vocabulario completo guardados en embeddings_output/")
            logger.info("   - tensor.tsv: Contiene los vectores de embeddings")
            logger.info("   - metadata.tsv: Contiene los tokens correspondientes")
            logger.info(f"   Se procesaron {len(vocab_words)} tokens en total")
            
            # Iniciar TensorBoard automáticamente después de la extracción
            logger.info("\nIniciando visualización...")
            self.tensorboard.start_tensorboard()
            
            # Salir del programa
            exit(0)
            
        except Exception as e:
            logger.error(f"❌ Error al extraer embeddings del vocabulario: {str(e)}")
            exit(1)

    def _visualize_embeddings(self):
        """Visualiza embeddings existentes en TensorBoard"""
        try:
            self.tensorboard.start_tensorboard()
            logger.info("✅ TensorBoard iniciado. Abre http://localhost:6006 en tu navegador")
        except Exception as e:
            logger.error(f"❌ Error al iniciar TensorBoard: {str(e)}")
