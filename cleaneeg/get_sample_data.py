import ftplib
import random
from pathlib import Path
from tqdm.notebook import tqdm


def is_dir(ftp: ftplib.FTP, path: str) -> bool:
    cwd = ftp.pwd()
    try:
        ftp.cwd(path)
        ftp.cwd(cwd)
        return True
    except ftplib.error_perm:
        return False


def download_remote(ftp: ftplib.FTP, remote_dir: str, local_dir: Path):
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        ftp.cwd(remote_dir)
    except ftplib.error_perm:
        return
    try:
        entries = list(ftp.mlsd())
    except (ftplib.error_perm, AttributeError):
        names = ftp.nlst()
        entries = [(name, {'type': 'dir' if is_dir(ftp, f"{remote_dir}/{name}") else 'file'})
                   for name in names]
    for name, info in tqdm(entries, desc=f"Scanning {Path(remote_dir).name}", leave=False):
        rpath = f"{remote_dir}/{name}"
        lpath = local_dir / name
        if info.get('type') == 'dir':
            download_remote(ftp, rpath, lpath)
        else:
            with open(lpath, 'wb') as f:
                ftp.retrbinary(f"RETR {rpath}", f.write)


def download_sample_data(ftp_host: str,
                         ftp_base: str,
                         local_base: Path,
                         num_subjects: int = 1):
    ftp = ftplib.FTP(ftp_host)
    ftp.login()
    ftp.cwd(ftp_base)
    subjects = ftp.nlst()
    if len(subjects) < num_subjects:
        raise ValueError(f"Found only {len(subjects)} subjects, asked for {num_subjects}")
    chosen = random.sample(subjects, num_subjects)
    print(f"→ Downloading {num_subjects} random subject(s): {chosen}\n")
    for subj in tqdm(chosen, desc="Subjects"):
        download_remote(ftp, f"{ftp_base}/{subj}", local_base / subj)
    ftp.quit()
    print("\n✅ Download complete. Data is in:", local_base.resolve())
