# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.
import logging
import os
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from git import (
    GitCommandError,
    InvalidGitRepositoryError,
    NoSuchPathError,
    Repo,
)

from judo import DESCRIPTION_CACHE_DIR


def acquire_lock(lock_path: Path, timeout: int = 60, poll_interval: float = 0.1) -> None:
    """Acquire a lock by creating a lock file atomically.

    Raises TimeoutError if the lock can't be acquired in `timeout` seconds.
    """
    start = time.time()
    while True:
        try:
            # open with O_CREAT | O_EXCL to create file atomically, fail if exists
            os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            return  # acquired
        except FileExistsError:
            if time.time() - start > timeout:
                raise TimeoutError(f"Timeout waiting for lock file {lock_path}") from None
            time.sleep(poll_interval)


def release_lock(lock_path: Path) -> None:
    """Remove the lock file to release the lock."""
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def download_and_extract_meshes(
    extract_root: str,
    repo: str = "bdaiinstitute/judo",
    asset_name: str = "meshes.zip",
    tag: str | None = None,
) -> None:
    """Downloads meshes.zip from the latest public GitHub release and extracts it."""
    extract_path = Path(extract_root).expanduser()
    meshes_path = extract_path / "meshes"
    lock_path = extract_path / ".meshes_download.lock"

    try:
        acquire_lock(lock_path)  # prevent race conditions resulting in multiple downloads

        # case: meshes already extracted
        if meshes_path.exists():
            return

        # fetch latest release info
        logging.info("Mesh assets not detected! Downloading assets now...")
        if tag is None:
            api_url = f"https://api.github.com/repos/{repo}/releases/latest"
        else:
            api_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
        response = requests.get(api_url)
        response.raise_for_status()
        release_data = response.json()

        # get the download URL for meshes.zip
        asset_url = None
        for asset in release_data.get("assets", []):
            if asset["name"] == asset_name:
                asset_url = asset["browser_download_url"]
                break
        if asset_url is None:
            raise ValueError(f"{asset_name} not found in latest release of {repo}.")

        # download and extract
        zip_path = meshes_path.with_suffix(".zip")
        meshes_path.mkdir(parents=True, exist_ok=True)
        with requests.get(asset_url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # extract the zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        if zip_path.exists():
            zip_path.unlink()  # remove the zip file after extraction

    finally:
        release_lock(lock_path)


@dataclass
class RepoInfo:
    """Information required to clone a description from a repository."""

    url: str
    description_path: str
    remote_model_path: str | None = None
    remote_meshes_path: str | None = None
    remote_xml_path: str | None = None
    commit_hash: str | None = None


KNOWN_DESCRIPTIONS = {
    # "spot": RepoInfo(
    #     url="", commit_hash=None, description_path="spot", remote_meshes_path="spot/assets", remote_xml_path="spot"
    # ),
    "cylinder_push": RepoInfo(
        url="https://github.com/bhung-bdai/judo_descriptions_test.git",
        commit_hash=None,
        description_path="judo_descriptions_test",
        remote_model_path="cylinder_push",
    ),
    "cartpole": RepoInfo(
        url="https://github.com/bhung-bdai/judo_descriptions_test.git",
        commit_hash=None,
        description_path="judo_descriptions_test",
        remote_model_path="cartpole",
    ),
    "fr3": RepoInfo(
        url="https://github.com/bhung-bdai/judo_descriptions_test.git",
        commit_hash=None,
        description_path="judo_descriptions_test",
        remote_model_path="franka_fr3",
    ),
    "leap_cube": RepoInfo(
        url="https://github.com/bhung-bdai/judo_descriptions_test.git",
        commit_hash=None,
        description_path="judo_descriptions_test",
        remote_model_path="leap_hand",
    ),
    "leap_cube_palm_down": RepoInfo(
        url="https://github.com/bhung-bdai/judo_descriptions_test.git",
        commit_hash=None,
        description_path="judo_descriptions_test",
        remote_model_path="leap_hand",
    ),
    "caltech_leap_cube": RepoInfo(
        url="https://github.com/bhung-bdai/judo_descriptions_test.git",
        commit_hash=None,
        description_path="judo_descriptions_test",
        remote_model_path="leap_hand",
    ),
}


def delete_assets_cache(cache_dir: Path | None = None) -> None:
    """Deletes the assets cache."""
    if cache_dir is None:
        cache_dir = Path(os.environ.get("JUDO_DESCRIPTION_CACHE_DIR", DESCRIPTION_CACHE_DIR)).expanduser()
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory {cache_dir} does not exist!")
    shutil.rmtree(cache_dir)


def get_description_keys() -> list[str]:
    """Returns a list of the keys of the known descriptions."""
    return list(KNOWN_DESCRIPTIONS.keys())


def add_known_description(description_name: str, description_info: RepoInfo) -> None:
    """Adds a new description via the RepoInfo to the known descriptions dictionary."""
    KNOWN_DESCRIPTIONS[description_name] = description_info


def get_description(description_info: RepoInfo, force: bool = False) -> Repo:
    """
    Clone a description from the provided information and return the path to the cloned repository.

    Args:
        description_info: The information about the description to clone.
        force: Forcefully redownload the description, regardless of whether or not it already exists.

    Returns:
        The path to the cloned description repository.
    """
    assert isinstance(description_info, RepoInfo), "description_info must be a RepoInfo object!"
    description_url = description_info.url
    description_path = Path(description_info.description_path)
    commit_hash = description_info.commit_hash

    # Find the directory hosting the cache
    cache_dir = Path(os.environ.get("JUDO_DESCRIPTION_CACHE_DIR", DESCRIPTION_CACHE_DIR)).expanduser()
    if commit_hash:
        description_path = description_path / commit_hash
    cache_path = cache_dir / description_path

    description_clone = None
    if force:
        if cache_path.exists():
            shutil.rmtree(cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Cloning description from {description_url} into {cache_path}")
        description_clone = Repo.clone_from(description_url, cache_path)
    else:
        try:
            description_clone = Repo(cache_path)
        except (InvalidGitRepositoryError, NoSuchPathError, GitCommandError):
            # Deletes the existing cache path for this repo if it exists and recreates it
            if cache_path.exists():
                shutil.rmtree(cache_path)
            cache_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Cloning description from {description_url} into {cache_path}")
            description_clone = Repo.clone_from(description_url, cache_path)

    if description_clone is None:
        raise FileNotFoundError(f"Failed to clone description from {description_url} into {cache_path}")

    if commit_hash:
        try:
            description_clone.git.checkout(commit_hash)
        except GitCommandError:
            description_clone.git.fetch("origin")
            description_clone.git.checkout(commit_hash)

    logging.info(f"Retrieved description from {description_url} to {description_clone.working_dir}")
    return description_clone


def retrieve_description_path_from_remote(description_name: str, force: bool = False) -> str:
    """Gets the path to a description from a remote repository, downloaded and stored in cache."""
    try:
        description_info = KNOWN_DESCRIPTIONS[description_name]
    except KeyError as e:
        raise ValueError(f"Description {description_name} not found in {get_description_keys()}!") from e
    description_clone = get_description(description_info, force=force)
    return f"{description_clone.working_dir}/{description_info.remote_model_path}"
