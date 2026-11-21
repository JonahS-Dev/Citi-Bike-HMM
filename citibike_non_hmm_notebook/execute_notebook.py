#!/usr/bin/env python3
"""Execute a notebook in a fresh IPython namespace and save its rich outputs.

This local runner needs no browser, Jupyter server or TCP ports. The notebook
also runs normally in JupyterLab/VS Code or with jupyter nbconvert --execute.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', nargs='?', default='citibike_non_hmm_baselines.ipynb')
    args = parser.parse_args()
    path = Path(args.notebook).resolve()
    os.chdir(path.parent)
    os.environ.setdefault('MPLBACKEND', 'module://matplotlib_inline.backend_inline')
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    from IPython.utils.capture import capture_output
    shell = TerminalInteractiveShell.instance()
    shell.display_formatter.active_types = ['text/plain', 'text/html', 'image/png', 'image/svg+xml']
    notebook = json.loads(path.read_text())
    start = time.perf_counter()
    count = 0
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        count += 1
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(source, store_history=True)
        cell['execution_count'] = result.execution_count
        outputs = []
        if captured.stdout:
            outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': captured.stdout})
        if captured.stderr:
            outputs.append({'output_type': 'stream', 'name': 'stderr', 'text': captured.stderr})
        for rich in captured.outputs:
            outputs.append({'output_type': 'display_data', 'data': rich.data, 'metadata': rich.metadata or {}})
        cell['outputs'] = outputs
        print(f'Executed code cell {count:02d} (notebook index {i})', flush=True)
        error = result.error_before_exec or result.error_in_exec
        if error:
            path.with_suffix('.failed.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n')
            raise RuntimeError(f'Notebook cell {i} failed: {error}') from error
    notebook['metadata']['language_info']['version'] = sys.version.split()[0]
    notebook['metadata']['execution'] = {'method': 'Sequential fresh-process IPython execution',
                                       'code_cells': count, 'all_cells_succeeded': True}
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n')
    print(f'Saved {count} executed code cells in {time.perf_counter()-start:.1f} seconds: {path.name}', flush=True)


if __name__ == '__main__':
    main()
