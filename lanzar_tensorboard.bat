@echo off
REM Script para lanzar TensorBoard en Windows usando el entorno virtual local
SETLOCAL
SET VENV_DIR=%~dp0territory\Scripts
IF EXIST "%VENV_DIR%\tensorboard.exe" (
    "%VENV_DIR%\tensorboard.exe" --logdir embeddings_output --port 6006
) ELSE IF EXIST "%VENV_DIR%\python.exe" (
    "%VENV_DIR%\python.exe" -m tensorboard.main --logdir embeddings_output --port 6006
) ELSE (
    echo TensorBoard no encontrado en el entorno virtual. Intenta instalarlo con: pip install tensorboard
    exit /b 1
)
ENDLOCAL
