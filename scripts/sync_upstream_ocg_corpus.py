"""Keep a verified, versioned copy of the public upstream OCG corpus."""
import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
PIN = ROOT / "build/reference-rag-pins"
STORE = ROOT / "build/reference-rag-corpora"
COMMIT = "41c6b8966263796b6548cc6b368c6bc06b04dc70"
API = "https://api.github.com/repos/coldiceh/ocg-ruling-assistant"
RAW = "https://raw.githubusercontent.com/coldiceh/ocg-ruling-assistant"
FILES = (
    "cards.json", "rulings.json", "qa-index.json", "qa-discovery-index.json",
    "evidence-index.json.gz", "ocg-rule-corpus.json", "official-responses.json",
    "rag-data-revision-manifest.json", "snapshot-meta.json",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def get(url, headers=None):
    return urlopen(Request(url, headers={"User-Agent": "upstream-ocg-corpus-sync", **(headers or {})}), timeout=90)


def api(path):
    with get(API + path) as response:
        return json.loads(response.read().decode("utf-8"))


def inventory(tree, folder):
    prefix = folder + "/" if folder else ""
    rows = {row["path"]: row for row in tree["tree"] if row["type"] == "blob"}
    return {name: rows[prefix + name] for name in FILES}


def blob(path, item):
    if path.stat().st_size != item["size"]:
        raise ValueError(f"Byte count mismatch for {path.name}")
    digest = hashlib.sha1(f"blob {item['size']}\0".encode("utf-8"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != item["sha"]:
        raise ValueError(f"Git blob mismatch for {path.name}")


def validate(folder, files):
    for name, item in files.items():
        blob(folder / name, item)
    manifest = read(folder / "rag-data-revision-manifest.json")
    baseline = read(PIN / "verified-data" / "rag-data-revision-manifest.json")
    for key in ("compilerAbi", "canonicalizationAbi"):
        if manifest.get(key) != baseline.get(key):
            raise ValueError(f"Upstream {key} differs from the pinned runtime")
    for item in manifest["sources"]:
        name = item["path"]
        if name not in FILES and name != "evidence-index.json":
            raise ValueError(f"Unexpected manifest source {name}")
        digest = hashlib.sha256()
        size = 0
        stream = (gzip.open(folder / "evidence-index.json.gz", "rb")
                  if name == "evidence-index.json" else (folder / name).open("rb"))
        with stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        if size != item["bytes"] or digest.hexdigest() != item["sha256"]:
            raise ValueError(f"Corpus manifest mismatch for {name}")
    counts = {name: len(read(folder / name)["records"]) for name in FILES
              if name not in {"evidence-index.json.gz", "rag-data-revision-manifest.json", "snapshot-meta.json"}}
    if any(counts[name] == 0 for name in ("cards.json", "rulings.json", "qa-index.json", "ocg-rule-corpus.json")):
        raise ValueError("A required corpus layer is empty")
    if (counts["cards.json"] != manifest["counts"]["cards"]
            or counts["qa-index.json"] != manifest["counts"]["qaRecords"]):
        raise ValueError("Corpus manifest record counts do not match")
    meta = read(folder / "snapshot-meta.json")
    discovery = read(folder / "qa-discovery-index.json")
    if discovery.get("sourceRevision") != meta.get("sourceRevision") or not discovery.get("complete"):
        raise ValueError("Q&A discovery index is incomplete or has a different source revision")
    return {"revision": manifest["revision"], "sourceRevision": meta["sourceRevision"], "counts": counts}


def download(commit, name, path, item):
    size = item["size"]
    windows_curl = Path("/mnt/d/Program Files/Git/mingw64/bin/curl.exe")
    if windows_curl.is_file():
        url = f"{RAW}/{commit}/data/{name}"
        command = [str(windows_curl), "--fail", "--silent", "--show-error",
                   "--max-time", "120"]
        chunk_size = 2 * 1024 * 1024
        if size <= chunk_size:
            with path.open("wb") as handle:
                subprocess.run([*command, url + "?snapshot=1"], stdout=handle, check=True)
            return

        parts = path.with_name(path.name + ".parts")
        parts.mkdir(exist_ok=True)

        def chunk(start):
            end = min(start + chunk_size, size) - 1
            part = parts / str(start)
            expected = end - start + 1
            if part.exists() and part.stat().st_size == expected:
                return part
            for _ in range(4):
                response = subprocess.run([*command, "--range", f"{start}-{end}",
                                           f"{url}?snapshot-offset={start}"],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if response.returncode == 0 and len(response.stdout) == expected:
                    part.write_bytes(response.stdout)
                    return part
            raise ValueError(f"Incomplete range for {name} at {start} (curl {response.returncode})")

        with ThreadPoolExecutor(max_workers=5) as pool, path.open("wb") as handle:
            for part in pool.map(chunk, range(0, size, chunk_size)):
                with part.open("rb") as block:
                    shutil.copyfileobj(block, handle)
        try:
            blob(path, item)
        except ValueError:
            shutil.rmtree(parts)
            path.unlink()
            raise
        shutil.rmtree(parts)
        return
    try:
        download_raw(commit, name, path, size)
    except (OSError, URLError, ValueError):
        payload = api("/git/blobs/" + item["sha"])
        if payload.get("encoding") != "base64" or payload.get("size") != size:
            raise ValueError(f"Unexpected GitHub blob payload for {name}")
        path.write_bytes(base64.b64decode(payload["content"]))


def download_raw(commit, name, path, size):
    url = f"{RAW}/{commit}/data/{name}"
    with get(url, {"Accept-Encoding": "identity"}) as response, path.open("wb") as out:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            out.write(block)
    if path.stat().st_size == size:
        return
    with path.open("wb") as out:
        for start in range(0, size, 2 * 1024 * 1024):
            end = min(start + 2 * 1024 * 1024, size) - 1
            with get(f"{url}?offset={start}", {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                    raise ValueError(f"Invalid range for {name} at {start}")
                block = response.read()
            if len(block) != end - start + 1:
                raise ValueError(f"Incomplete range for {name} at {start}")
            out.write(block)


def smoke_test(folder):
    backend = PIN / "backend" / "ragEvidenceRetriever.mjs"
    node = shutil.which("node") or shutil.which("node.exe")
    if node is None:
        windows_node = Path("/mnt/c/Program Files/nodejs/node.exe")
        if windows_node.is_file():
            node = str(windows_node)
    if node is None:
        raise RuntimeError("Node is required to verify the upstream corpus can load")
    paths = [str(backend), str(folder)]
    if node.endswith(".exe") and shutil.which("wslpath"):
        paths = [subprocess.check_output(["wslpath", "-w", path], encoding="utf-8").strip()
                 for path in paths]
    js = (
        'import { pathToFileURL } from "node:url";'
        'const {loadRagData}=await import(pathToFileURL(process.argv[1]));'
        'const data=await loadRagData(process.argv[2]);'
        'if (!data.cards.length || !data.records.length || !data.qaRecords.length)'
        ' throw new Error("Upstream runtime loaded empty data");'
    )
    subprocess.run([node, "--input-type=module", "-e", js, *paths], check=True, timeout=120)


def save_pointer(commit, folder, metadata):
    data = {"commit": commit, "dataDir": folder.relative_to(ROOT).as_posix(), **metadata}
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=STORE, delete=False) as pending:
        json.dump(data, pending, ensure_ascii=False, indent=2)
        pending.write("\n")
    os.replace(pending.name, STORE / "current.json")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("init", "status", "refresh"))
    operation = parser.parse_args().operation
    pointer = STORE / "current.json"
    if operation == "init" and not pointer.exists():
        STORE.mkdir(parents=True, exist_ok=True)
        folder = PIN / "verified-data"
        current = save_pointer(COMMIT, folder, validate(folder, inventory(read(PIN / "tree.json"), "data")))
        print(json.dumps(current, ensure_ascii=False))
        return
    current = read(pointer)
    if operation != "refresh":
        print(json.dumps(current, ensure_ascii=False))
        return
    head = api("/commits/main")
    commit = head["sha"]
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("Invalid upstream Git commit")
    if commit == current["commit"]:
        print("No upstream commit change")
        return
    tree = api("/git/trees/" + head["commit"]["tree"]["sha"])
    data_tree = next(row["sha"] for row in tree["tree"] if row["path"] == "data" and row["type"] == "tree")
    files = inventory(api("/git/trees/" + data_tree), "")
    current_dir = ROOT / current["dataDir"]
    if all((current_dir / name).stat().st_size == item["size"] for name, item in files.items()):
        try:
            for name, item in files.items():
                blob(current_dir / name, item)
        except ValueError:
            pass
        else:
            print("No upstream corpus content change")
            return
    destination = STORE / commit
    if not destination.exists():
        staged = STORE / (".staging-" + commit)
        staged.mkdir(exist_ok=True)
        for name, item in files.items():
            pending = staged / name
            try:
                blob(pending, item)
                continue
            except (OSError, ValueError):
                pass
            previous = current_dir / name
            try:
                blob(previous, item)
            except (OSError, ValueError):
                print(f"Downloading {name}", flush=True)
                download(commit, name, pending, item)
            else:
                shutil.copyfile(previous, pending)
        metadata = validate(staged, files)
        smoke_test(staged)
        staged.rename(destination)
    else:
        metadata = validate(destination, files)
        smoke_test(destination)
    print(json.dumps(save_pointer(commit, destination, metadata), ensure_ascii=False))


if __name__ == "__main__":
    main()
