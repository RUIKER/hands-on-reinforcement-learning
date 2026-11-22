"""Fetch the official openpi LIBERO checkpoint with resumable GCS downloads."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


base = 'https://storage.googleapis.com/openpi-assets'
prefix = 'checkpoints/pi05_libero/'
target = Path(sys.argv[1])
manifest_path = target.with_suffix('.manifest.xml')
if not manifest_path.exists():
    data = subprocess.check_output(['curl', '-fsSL', '--retry', '2', '--max-time', '25', f'{base}?prefix={prefix}'])
    ET.fromstring(data)
    manifest_path.write_bytes(data)
manifest = ET.parse(manifest_path)


def fetch(entry):
    key = entry.findtext('{*}Key')
    generation = entry.findtext('{*}Generation')
    size = int(entry.findtext('{*}Size'))
    path = target / key.removeprefix(prefix)
    if path.is_file() and path.stat().st_size == size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    if not part.exists() or part.stat().st_size != size:
        subprocess.run(['curl', '-fsSL', '--retry', '1', '--connect-timeout', '12',
                        '--max-time', '1500', '-C', '-', '-o', str(part),
                        f'{base}/{key}?generation={generation}'], check=True)
    if part.stat().st_size != size:
        raise ValueError(f'Unexpected size for {key}')
    part.replace(path)
    print(f'{size} {key}', flush=True)


with ThreadPoolExecutor(max_workers=6) as pool:
    list(pool.map(fetch, manifest.findall('{*}Contents')))
