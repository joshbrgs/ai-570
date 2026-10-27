# ai-570 — SeeFood (Keras Functional API)

A multi-output Keras model (hot dog / not hot dog + condiment recommendation) built with the
Functional API. Development happens in JupyterLab running in Docker, so the only thing you need
installed is **Docker Desktop** (or Docker Engine + Compose on Linux).

## Quick start

```bash
docker compose up -d --build
```

Open **http://localhost:8888/lab?token=seefood**.

The whole repo is mounted at `/workspace`, so notebooks you create or edit, and any files your
code writes (plots, saved models), show up directly in the repo on your machine.

```bash
docker compose logs -f jupyter   # view server logs
docker compose down              # stop the server
```

## Data (Galaxy Zoo 2)

The data isn't stored in git. [DVC](https://dvc.org) tracks it straight from its public sources
(the `data/raw/*.dvc` files record each URL and checksum), so there's no shared bucket to set up.
After cloning, run this on your machine (needs [uv](https://docs.astral.sh/uv/), ~8 GB free disk):

```bash
uv sync --all-groups
uv run dvc update data/raw/*.dvc   # downloads the raw files from source into data/raw/ (~3.6 GB)
uv run dvc repro                   # decompresses them into data/*.csv and data/images/
```

- **Use `dvc update`, not `dvc pull`.** `pull` downloads from a DVC remote (a storage bucket), and
  this project deliberately doesn't have one. `update` downloads each file from its original URL.
- **Don't run `dvc init`.** The repo is already set up for DVC (`.dvc/` is committed).
- **Labels only?** The notebooks read the `.csv.gz` files in `data/raw/` directly, so if you don't
  need images, run `uv run dvc update $(ls data/raw/*.dvc | grep -v images_gz2)` and skip the
  3.4 GB image zip and `dvc repro`.
- After the first download, `data/raw/images_gz2.zip.dvc` gains the zip's checksum. That change
  is harmless, so there's no need to commit it.
- **`SSL: CERTIFICATE_VERIFY_FAILED` on macOS?** Your Python (from the python.org installer) has
  no root certificates. Run `/Applications/Python 3.12/Install Certificates.command` once.

**Google Colab:** Colab's disk is wiped between sessions, so this downloads again each time.
`scripts/extract_data.py` only needs the standard library, so run it directly instead of through
`uv`:

```python
!git clone https://github.com/joshbrgs/ai-570.git
%cd ai-570
!pip install -q dvc
!dvc update data/raw/*.dvc
!python scripts/extract_data.py
import sys; sys.path.append("/content/ai-570")   # makes `from gz2.data import load_galaxies` work
```

| Path | Contents |
|------|----------|
| `data/gz2_hart16.csv` | GZ2 Table 1, Hart et al. (2016) debiased morphology. The main label table |
| `data/gz2sample.csv`, `data/zoo2*.csv` | Other GZ2 tables from [data.galaxyzoo.org](https://data.galaxyzoo.org) |
| `data/gz2_filename_mapping.csv` | `objid` → `asset_id`; the image for a galaxy is `data/images/<asset_id>.jpg` |
| `data/images/` | 243,434 GZ2 images ([Zenodo 3565489](https://doi.org/10.5281/zenodo.3565489)) |

Each `.dvc` file records its file's checksum. If a source ever changes upstream, `dvc update`
downloads the new version and the new checksum shows up as a `git diff` on that `.dvc` file.

**Citation:** cite Willett et al. (2013, MNRAS 435, 2835), Hart et al. (2016, MNRAS 461, 3663) and
the Zenodo dataset DOI `10.5281/zenodo.3565489`.

## Layout

| Path | Purpose |
|------|---------|
| `main.py` | Model definition (`build_model()`); run it to print the summary and write `seefood2.png` |
| `notebooks/` | Experiment notebooks. The repo root is on `PYTHONPATH`, so `from main import build_model` works |
| `Dockerfile` / `compose.yaml` | Python 3.12 + TensorFlow/Keras + JupyterLab + graphviz |

Run a script inside the container:

```bash
docker compose exec jupyter python main.py
```

## Configuration

Copy `.env.example` to `.env` to override defaults (`.env` is gitignored):

- `JUPYTER_TOKEN`: login token (default `seefood`). The server only listens on localhost.
- `JUPYTER_PORT`: host port (default `8888`). Change it if 8888 is already taken.
- `UID` / `GID`: **Linux only.** Set them to the output of `id -u` / `id -g` so files the container
  creates belong to you. Docker Desktop on macOS/Windows handles this for you.

## Adding dependencies

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and pinned in `uv.lock`. The image
installs from the lockfile and refuses to build if the lock is out of date.

If you have uv installed:

```bash
uv add <package>                    # runtime dependency
uv add --group notebook <package>   # notebook-only tooling
```

If you don't, run uv inside the container (it writes `pyproject.toml` and `uv.lock` into the repo):

```bash
docker compose exec jupyter uv add <package>
```

Either way, rebuild afterward and commit both `pyproject.toml` and `uv.lock`:

```bash
docker compose up -d --build
```

## Notebook hygiene

Clear outputs before committing (in JupyterLab: *Edit → Clear Outputs of All Cells*) to keep diffs
readable and the repo small.
