"""Batch OpenFace2 feature extraction."""

import os
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
import shlex
import uuid

def run(cmd, log_file=None):
    """Run a shell command, optionally logging output."""
    if log_file:
        with open(log_file, "w") as f:
            result = subprocess.run(cmd, shell=True, executable="/bin/bash",
                           stdout=f, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd)
    else:
        result = subprocess.run(cmd, shell=True, executable="/bin/bash")
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)

def find_all_videos(videos_base):
    """Recursively find all video files in the directory."""
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv', '.flv', '.wmv',
                        '.AVI', '.MP4', '.MOV']
    video_files = []
    for ext in video_extensions:
        video_files.extend(videos_base.rglob(f'*{ext}'))
    return sorted(set(video_files))

def main():
    SUFFIX = "..."
    BASE_SHARED = Path("...")
    videos_base = BASE_SHARED / "videos"
    temp_dir = BASE_SHARED / "OpenFace2" / f"temp_{SUFFIX}"
    results_root = BASE_SHARED / f"openface2_results_{SUFFIX}"
    sif_path = BASE_SHARED / "OpenFace2" / "openface.sif"
    MAX_ATTEMPTS = 3

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(results_root, exist_ok=True)
    
    video_files = find_all_videos(videos_base)
    if not video_files:
        print("No video files found. Exiting.")
        return

    failed_videos = []
    
    for video_path in tqdm(video_files, desc="Total videos", unit="video"):
        video_path = Path(video_path)
        name = video_path.stem
        rel_path = video_path.relative_to(videos_base)
        result_dir = results_root / rel_path.parent
        os.makedirs(result_dir, exist_ok=True)
        output_dir = result_dir / f"{name}_processed"
        if output_dir.exists() and list(output_dir.glob("*.csv")):
            print(f"Skipping {rel_path}, already processed.")
            continue

        random_name = str(uuid.uuid4())
        temp_avi = temp_dir / f"{random_name}.avi"

        if temp_avi.exists():
            temp_avi.unlink()

        video_quoted         = shlex.quote(str(video_path))
        temp_avi_quoted      = shlex.quote(str(temp_avi))
        base_shared_quoted   = shlex.quote(str(BASE_SHARED))
        sif_path_quoted      = shlex.quote(str(sif_path))

        temp_avi_inside      = Path(f"/mnt/data/OpenFace2/temp_{SUFFIX}") / f"{random_name}.avi"
        temp_avi_inside_quoted = shlex.quote(str(temp_avi_inside))
        dst_path_inside      = Path(f"/mnt/data/openface2_results_{SUFFIX}") / rel_path.parent / f"{name}_processed"
        dst_path_str         = str(dst_path_inside)  # intentionally not quoted (used inside bash -c)
        work_dir_inside      = Path(f"/mnt/data/OpenFace2/temp_{SUFFIX}")
        work_dir_quoted      = shlex.quote(str(work_dir_inside))

        steps = [
            ("FFmpeg convert",
             f"conda run -n openface ffmpeg -y -i {video_quoted} "
             f"-c:v mjpeg -q:v 2 -pix_fmt yuvj420p {temp_avi_quoted}"),
            ("OpenFace extraction",
             f"singularity exec -B {base_shared_quoted}:/mnt/data {sif_path_quoted} "
             f"bash -c 'cd {work_dir_quoted} && FeatureExtraction -f {temp_avi_inside_quoted}'"),
            ("Copy CSV",
             f"singularity exec -B {base_shared_quoted}:/mnt/data {sif_path_quoted} "
             f"bash -c 'mkdir -p \"{dst_path_str}\" && cp {work_dir_quoted}/processed/*.csv \"{dst_path_str}/\"'"),
            ("Clear processed",
             f"singularity exec -B {base_shared_quoted}:/mnt/data {sif_path_quoted} "
             f"bash -c 'rm -rf {work_dir_quoted}/processed'"),
        ]

        success = False

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                for label, cmd in tqdm(steps, desc=f"{rel_path} (attempt {attempt})", leave=False, unit="step"):
                    log_file = temp_dir / f"{random_name}_{label.replace(' ', '_')}_attempt{attempt}.log"
                    run(cmd, log_file=log_file)
                success = True
                break
            except subprocess.CalledProcessError:
                if attempt < MAX_ATTEMPTS:
                    tqdm.write(f"Attempt {attempt} failed for {rel_path}. Retrying...")
                    if temp_avi.exists():
                        temp_avi.unlink()
                else:
                    tqdm.write(f"Failed to process {rel_path} after {MAX_ATTEMPTS} attempts. Skipping.")
                    failed_videos.append(str(rel_path))
        
        if not success:
            continue

    print("\n" + "="*60)
    if failed_videos:
        print(f"Processing completed with {len(failed_videos)} failures:")
        for vid in failed_videos:
            print(f"  - {vid}")
    else:
        print("All videos processed successfully with mirrored folder structure.")
    print("="*60)

if __name__ == "__main__":
    main()
