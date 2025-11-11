# training/github_sync.py
from __future__ import annotations
import base64
import json
import os
from typing import Optional, Tuple

import requests

API_BASE = "https://api.github.com"

class GHError(RuntimeError):
    pass

def _headers(token: str) -> dict:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "BactAI-D-GitClient"
    }

def _repo_parts(repo_full: str) -> Tuple[str, str]:
    if "/" not in repo_full:
        raise GHError(f"Invalid repo name '{repo_full}'. Expected 'owner/repo'.")
    owner, repo = repo_full.split("/", 1)
    return owner, repo

def get_default_branch_sha(token: str, repo_full: str, branch: str) -> str:
    owner, repo = _repo_parts(repo_full)
    url = f"{API_BASE}/repos/{owner}/{repo}/git/ref/heads/{branch}"
    r = requests.get(url, headers=_headers(token), timeout=20)
    if r.status_code != 200:
        raise GHError(f"Fetch branch '{branch}' failed: {r.status_code} - {r.text}")
    return r.json()["object"]["sha"]

def get_file_sha_if_exists(token: str, repo_full: str, path: str, ref: str) -> Optional[str]:
    owner, repo = _repo_parts(repo_full)
    url = f"{API_BASE}/repos/{owner}/{repo}/contents/{path}"
    r = requests.get(url, headers=_headers(token), params={"ref": ref}, timeout=20)
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        raise GHError(f"Get contents failed: {r.status_code} - {r.text}")
    return r.json().get("sha")

def create_branch(token: str, repo_full: str, new_branch: str, from_sha: str) -> None:
    owner, repo = _repo_parts(repo_full)
    url = f"{API_BASE}/repos/{owner}/{repo}/git/refs"
    payload = {"ref": f"refs/heads/{new_branch}", "sha": from_sha}
    r = requests.post(url, headers=_headers(token), json=payload, timeout=20)
    if r.status_code not in (201, 422):  # 422 if already exists
        raise GHError(f"Create branch failed: {r.status_code} - {r.text}")

def put_file(
    token: str,
    repo_full: str,
    path: str,
    content_text: str,
    message: str,
    branch: str,
    committer_name: str,
    committer_email: str,
) -> dict:
    """
    Create or update a file via PUT /contents/{path}
    """
    owner, repo = _repo_parts(repo_full)
    url = f"{API_BASE}/repos/{owner}/{repo}/contents/{path}"

    current_sha = get_file_sha_if_exists(token, repo_full, path, ref=branch)
    payload = {
        "message": message,
        "content": base64.b64encode(content_text.encode("utf-8")).decode("utf-8"),
        "branch": branch,
        "committer": {"name": committer_name, "email": committer_email},
    }
    if current_sha:
        payload["sha"] = current_sha

    r = requests.put(url, headers=_headers(token), json=payload, timeout=25)
    if r.status_code not in (200, 201):
        raise GHError(f"Commit failed: {r.status_code} - {r.text}")
    return r.json()

def open_pull_request(
    token: str,
    repo_full: str,
    title: str,
    head_branch: str,
    base_branch: str,
    body: str = ""
) -> dict:
    owner, repo = _repo_parts(repo_full)
    url = f"{API_BASE}/repos/{owner}/{repo}/pulls"
    payload = {"title": title, "head": head_branch, "base": base_branch, "body": body}
    r = requests.post(url, headers=_headers(token), json=payload, timeout=20)
    if r.status_code != 201:
        raise GHError(f"Open PR failed: {r.status_code} - {r.text}")
    return r.json()
