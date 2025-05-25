import subprocess
import webbrowser
from pathlib import Path
import logging
from time import sleep
import os

from src.utils.config import Config

logger = logging.getLogger(__name__)

class TensorBoardManager:
    """Gestiona la visualización con TensorBoard"""
    
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(self.config.get('output_dir', 'embeddings_output'))
        self.output_dir.mkdir(exist_ok=True)
        self.process = None

    def prepare_projector_config(self):
        """Prepara el archivo de configuración para TensorBoard"""
        if not (self.output_dir / "tensor.tsv").exists() or not (self.output_dir / "metadata.tsv").exists():
            raise FileNotFoundError("No se encontraron archivos de embeddings. Extrae embeddings primero.")
            
        # Crear archivo de configuración
        config_content = '''embeddings {
  tensor_name: "embeddings"
  tensor_path: "tensor.tsv"
  metadata_path: "metadata.tsv"
}'''
        with open(self.output_dir / "projector_config.pbtxt", 'w') as f:
            f.write(config_content)

    def start_tensorboard(self):
        """Inicia TensorBoard con la configuración actual, detectando el entorno automáticamente"""
        try:
            self.prepare_projector_config()
            if self.process:
                logger.info("TensorBoard ya está corriendo")
                return
            port = self.config.get('tensorboard', {}).get('default_port', 6006)
            # Detección automática de comando para Windows
            tb_cmd = None
            if os.name == 'nt':
                # Buscar tensorboard.exe en venv
                venv_tb = Path('territory') / 'Scripts' / 'tensorboard.exe'
                venv_py = Path('territory') / 'Scripts' / 'python.exe'
                if venv_tb.exists():
                    tb_cmd = [str(venv_tb), '--logdir', str(self.output_dir), '--port', str(port)]
                elif venv_py.exists():
                    tb_cmd = [str(venv_py), '-m', 'tensorboard.main', '--logdir', str(self.output_dir), '--port', str(port)]
                else:
                    tb_cmd = ['tensorboard', '--logdir', str(self.output_dir), '--port', str(port)]
            else:
                tb_cmd = ['tensorboard', '--logdir', str(self.output_dir), '--port', str(port)]
            logger.info(f"Iniciando TensorBoard en el puerto {port}...")
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            try:
                self.process = subprocess.Popen(
                    tb_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    startupinfo=startupinfo
                )
            except FileNotFoundError:
                logger.error("❌ TensorBoard no está instalado o no se encuentra en el entorno. Por favor, instala TensorBoard con 'pip install tensorboard'.")
                return
            sleep(2)
            if self.process.poll() is not None:
                stderr = self.process.stderr.read().decode()
                raise RuntimeError(f"TensorBoard falló al iniciar: {stderr}")
            url = f"http://localhost:{port}/#projector"
            logger.info(f"\n✨ TensorBoard iniciado en {url}")
            logger.info("Abriendo navegador...")
            webbrowser.open(url)
        except FileNotFoundError as e:
            logger.error(f"❌ {str(e)}")
        except Exception as e:
            logger.error(f"❌ Error al iniciar TensorBoard: {str(e)}")
            if self.process:
                self.process.kill()
                self.process = None
