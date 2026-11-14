"""Launch the local PCB app with the selected on-demand model backend."""
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    source = str(root / 'src')
    os.environ['PYTHONPATH'] = source + os.pathsep + os.environ.get('PYTHONPATH', '')
    sys.path.insert(0, source)
    from circuitcare.vision.weight_paths import resolve_pcb_weights_root
    os.environ['CIRCUITCARE_PCB_WEIGHTS'] = str(resolve_pcb_weights_root(root / "apps/pcb-component-recognition"))
    backend = os.environ.setdefault('CIRCUITCARE_LLM_BACKEND', 'mlx' if sys.platform == 'darwin' else 'ollama')
    configured_python = os.environ.get('CIRCUITCARE_APP_PYTHON')
    weights_root = Path(os.environ['CIRCUITCARE_PCB_WEIGHTS'])
    weight_project_root = weights_root.parent.parent.parent
    candidates = [Path(configured_python)] if configured_python else []
    candidates.extend([
        Path(sys.executable),
        weight_project_root / '.venv-component-train/bin/python',
    ])
    required_modules = ['torch', 'ultralytics', 'streamlit'] + (['mlx_vlm'] if backend == 'mlx' else [])
    probe = "import importlib.util, sys; missing=[name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]; sys.exit(1 if missing else 0)"
    app_interpreter = None
    # Keep the venv launcher path intact. Resolving ``.venv/bin/python`` to the
    # Homebrew framework binary drops the venv context on macOS, so module
    # probes (torch/ultralytics/streamlit/mlx_vlm) incorrectly report missing.
    for candidate in dict.fromkeys(path.expanduser().absolute() for path in candidates):
        if (candidate.is_file()
                and subprocess.run([str(candidate), '-c', probe, *required_modules],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   check=False).returncode == 0):
            app_interpreter = candidate
            break
    if app_interpreter is None:
        raise SystemExit(
            'PCB 실행 Python 환경에 torch, ultralytics, streamlit'
            + (', mlx_vlm' if backend == 'mlx' else '')
            + '이 필요합니다. CIRCUITCARE_APP_PYTHON으로 올바른 환경을 지정하세요.'
        )
    os.environ['CIRCUITCARE_APP_PYTHON'] = str(app_interpreter)
    if backend == 'mlx':
        if 'CIRCUITCARE_MLX_PYTHON' not in os.environ:
            os.environ['CIRCUITCARE_MLX_PYTHON'] = str(app_interpreter)
        os.environ.setdefault('CIRCUITCARE_MLX_MODEL', 'mlx-community/Qwen3.5-9B-MLX-4bit')
        mlx_interpreter = Path(os.environ['CIRCUITCARE_MLX_PYTHON']).expanduser().absolute()
        if not mlx_interpreter.is_file():
            raise SystemExit('MLX Python environment is missing. Set CIRCUITCARE_MLX_PYTHON.')
        if subprocess.run([str(mlx_interpreter), '-c', probe, 'mlx_vlm'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          check=False).returncode != 0:
            raise SystemExit('CIRCUITCARE_MLX_PYTHON must provide mlx_vlm.')
        os.environ['CIRCUITCARE_MLX_PYTHON'] = str(mlx_interpreter)
    os.execv(str(app_interpreter), [str(app_interpreter), '-m', 'streamlit', 'run',
        str(root / 'src/circuitcare/ui.py'), '--server.address', '127.0.0.1',
        '--server.port', '8765', '--server.headless', 'true'])


if __name__ == '__main__':
    main()
