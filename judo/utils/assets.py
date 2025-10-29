# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from git import (
    GitCommandError,
    InvalidGitRepositoryError,
    NoSuchPathError,
    Repo,
)

from judo import DESCRIPTION_CACHE_DIR


@dataclass(frozen=True)
class RepoInfo:
    """Information required to clone a description from a repository."""

    url: str
    description_path: str
    remote_model_path: str
    remote_model_file: str  # File name representing the model
    remote_sim_file: str | None = None  # Optional simulation model in case it's not the same as the model
    commit_hash: str | None = None


KNOWN_DESCRIPTIONS = {
    "cylinder_push": RepoInfo(
        url="https://github.com/bhung-bdai/judo_descriptions_test.git",
        commit_hash=None,
        description_path="judo_descriptions_test",
        remote_model_path="cylinder_push",
        remote_model_file="cylinder_push.xml",
    ),
    "cartpole": RepoInfo(
        url="https://github.com/bhung-bdai/judo_descriptions_test.git",
        commit_hash=None,
        description_path="judo_descriptions_test",
        remote_model_path="cartpole",
        remote_model_file="cartpole.xml",
    ),
    "fr3": RepoInfo(
        url="https://github.com/bhung-bdai/judo_descriptions_test.git",
        commit_hash=None,
        description_path="judo_descriptions_test",
        remote_model_path="franka_fr3",
        remote_model_file="fr3_pick.xml",
    ),
    "leap_cube": RepoInfo(
        url="https://github.com/bhung-bdai/judo_descriptions_test.git",
        commit_hash=None,
        description_path="judo_descriptions_test",
        remote_model_path="leap_hand",
        remote_model_file="leap_cube.xml",
    ),
    "leap_cube_palm_down": RepoInfo(
        url="https://github.com/bhung-bdai/judo_descriptions_test.git",
        commit_hash=None,
        description_path="judo_descriptions_test",
        remote_model_path="leap_hand",
        remote_model_file="leap_cube_palm_down.xml",
        remote_sim_file="leap_cube_palm_down_sim.xml",
    ),
    "caltech_leap_cube": RepoInfo(
        url="https://github.com/bhung-bdai/judo_descriptions_test.git",
        commit_hash=None,
        description_path="judo_descriptions_test",
        remote_model_path="leap_hand",
        remote_model_file="caltech_leap_cube.xml",
        remote_sim_file="caltech_leap_cube_sim.xml",
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
    """Gets the path to a description from a remote repository, downloaded and stored in cache.

    Args:
        description_name: The name of the description to retrieve.
        force: Forcefully redownload the description, regardless of whether or not it already exists.

    Returns:
        The path to the description repository in the local cache.
    """
    try:
        description_info = KNOWN_DESCRIPTIONS[description_name]
    except KeyError as e:
        raise ValueError(f"Description {description_name} not found in {get_description_keys()}!") from e
    description_clone = get_description(description_info, force=force)
    return f"{description_clone.working_dir}/{description_info.remote_model_path}"


def retrieve_model_from_remote(description_name: str, force: bool = False) -> tuple[str, str | None]:
    """Gets the path to a model (and simulation file, if available) in a remote description, stored in cache.

    Args:
        description_name: The name of the description to retrieve.
        force: Forcefully redownload the description, regardless of whether or not it already exists.

    Returns:
        The path to the description repository in the local cache.
    """
    try:
        description_info = KNOWN_DESCRIPTIONS[description_name]
    except KeyError as e:
        raise ValueError(f"Description {description_name} not found in {get_description_keys()}!") from e
    description_model_path = retrieve_description_path_from_remote(description_name, force=force)
    description_model_file = f"{description_model_path}/{description_info.remote_model_file}"

    if description_info.remote_sim_file is not None:
        return (description_model_file, f"{description_model_path}/{description_info.remote_sim_file}")
    return (description_model_file, None)
