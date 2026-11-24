#!/usr/bin/env python3
"""
scripts/tobid_oauth_login.py - ToBid MCP OAuth 2.0 (DCR + PKCE) 本機登入輔助工具

透過 ToBid 公開 OAuth Discovery (https://api.tobid.tw/.well-known/oauth-authorization-server)
進行 Dynamic Client Registration 與 PKCE S256 授權流程，取得 access_token 與 refresh_token
後以 0600 權限寫入 cache/tender/oauth_token.json（不會於終端機輸出憑證原文）。

用法:
  .venv/bin/python scripts/tobid_oauth_login.py
  .venv/bin/python scripts/tobid_oauth_login.py --upload-remote 10.9.0.99:/home/david/stock-underdog-ml/cache/tender/oauth_token.json --restart-service stock-ml-api
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import http.server
import json
import os
import secrets
import shlex
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from typing import Any, Dict, Optional, Tuple
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data.tender_fetcher import TenderService

DISCOVERY_URL = "https://api.tobid.tw/.well-known/oauth-authorization-server"
RESOURCE_URI = "https://api.tobid.tw/mcp"
DEFAULT_SCOPE = "tobid"


def generate_pkce_pair() -> Tuple[str, str]:
    """產生 RFC 7636 PKCE code_verifier 與 S256 code_challenge"""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def fetch_oauth_metadata(timeout: float = 10.0) -> Dict[str, Any]:
    resp = requests.get(DISCOVERY_URL, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("Invalid OAuth discovery metadata")
    return data


def register_dynamic_client(registration_endpoint: str, redirect_uri: str, timeout: float = 10.0) -> Dict[str, Any]:
    payload = {
        "client_name": "stock-underdog-ml",
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
        "scope": DEFAULT_SCOPE,
    }
    resp = requests.post(
        registration_endpoint,
        json=payload,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict) or not data.get("client_id"):
        raise ValueError("OAuth client registration failed: missing client_id")
    return data


def exchange_authorization_code(
    token_endpoint: str,
    *,
    code: str,
    redirect_uri: str,
    client_id: str,
    code_verifier: str,
    client_secret: Optional[str] = None,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": code_verifier,
        "resource": RESOURCE_URI,
    }
    if client_secret:
        form["client_secret"] = client_secret
    resp = requests.post(
        token_endpoint,
        data=form,
        headers={"Accept": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict) or not data.get("access_token"):
        raise ValueError("OAuth token exchange failed: missing access_token")
    return data


def save_token_bundle(
    token_data: Dict[str, Any],
    *,
    client_id: str,
    client_secret: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    path = output_path or TenderService._get_token_file_path()
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    state: Dict[str, Any] = {
        "access_token": token_data["access_token"],
        "token_type": token_data.get("token_type", "Bearer"),
        "scope": token_data.get("scope", DEFAULT_SCOPE),
        "client_id": client_id,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if token_data.get("refresh_token"):
        state["refresh_token"] = token_data["refresh_token"]
    if client_secret:
        state["client_secret"] = client_secret
    if token_data.get("expires_in") is not None:
        try:
            state["expires_at"] = time.time() + float(token_data["expires_in"])
        except (TypeError, ValueError):
            pass

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, path)
    return path


def run_local_oauth_flow(
    *,
    timeout_seconds: int = 180,
    open_browser: bool = True,
    output_path: Optional[str] = None,
) -> str:
    meta = fetch_oauth_metadata()
    auth_endpoint = meta["authorization_endpoint"]
    token_endpoint = meta["token_endpoint"]
    reg_endpoint = meta["registration_endpoint"]

    result_box: Dict[str, Optional[str]] = {"code": None, "error": None}
    expected_state = secrets.token_urlsafe(24)
    done_event = threading.Event()

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/callback":
                self.send_response(404)
                self.end_headers()
                return

            params = urllib.parse.parse_qs(parsed.query)
            state_val = (params.get("state") or [""])[0]
            err_val = (params.get("error") or [""])[0]
            code_val = (params.get("code") or [""])[0]

            if state_val != expected_state:
                result_box["error"] = "state_mismatch"
                body = "<h1>OAuth State Mismatch</h1><p>授權驗證失敗，請關閉分頁重試。</p>"
                status = 400
            elif err_val:
                result_box["error"] = err_val
                body = f"<h1>OAuth Error</h1><p>{html.escape(err_val)}</p>"
                status = 400
            elif not code_val:
                result_box["error"] = "missing_code"
                body = "<h1>Missing Code</h1><p>未收到授權碼。</p>"
                status = 400
            else:
                result_box["code"] = code_val
                body = "<h1>ToBid 授權成功</h1><p>憑證已寫入本機快取，您可以關閉此瀏覽器分頁。</p>"
                status = 200

            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            done_event.set()

        def log_message(self, format, *args):
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), CallbackHandler)
    port = server.server_address[1]
    redirect_uri = f"http://127.0.0.1:{port}/callback"

    client_info = register_dynamic_client(reg_endpoint, redirect_uri)
    client_id = client_info["client_id"]
    client_secret = client_info.get("client_secret")

    verifier, challenge = generate_pkce_pair()
    auth_params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": DEFAULT_SCOPE,
        "state": expected_state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "resource": RESOURCE_URI,
    }
    auth_url = f"{auth_endpoint}?{urllib.parse.urlencode(auth_params)}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        print(f"請於瀏覽器完成 ToBid 登入授權（逾時 {timeout_seconds} 秒）：\n{auth_url}")
        if open_browser:
            webbrowser.open(auth_url)

        if not done_event.wait(timeout=timeout_seconds):
            raise TimeoutError(f"等待瀏覽器 OAuth 回調逾時（{timeout_seconds} 秒）")
        if result_box["error"]:
            raise RuntimeError(f"ToBid OAuth 授權失敗: {result_box['error']}")
        assert result_box["code"] is not None

        token_data = exchange_authorization_code(
            token_endpoint,
            code=result_box["code"],
            redirect_uri=redirect_uri,
            client_id=client_id,
            code_verifier=verifier,
            client_secret=client_secret,
        )
        saved_path = save_token_bundle(
            token_data,
            client_id=client_id,
            client_secret=client_secret,
            output_path=output_path,
        )
        has_refresh = bool(token_data.get("refresh_token"))
        expires_in = token_data.get("expires_in", "unknown")
        print(f"✅ ToBid OAuth 憑證已儲存：{saved_path} (refresh_token={has_refresh}, expires_in={expires_in}s)")
        return saved_path
    finally:
        server.shutdown()
        server.server_close()


def upload_to_remote(local_path: str, remote_spec: str, restart_service: Optional[str] = None) -> None:
    if ":" not in remote_spec:
        raise ValueError("--upload-remote 格式應為 host:/remote/path/oauth_token.json")
    host, remote_path = remote_spec.split(":", 1)
    remote_dir = os.path.dirname(remote_path)
    repo_dir = os.path.dirname(os.path.dirname(remote_dir)) if remote_dir.endswith("/tender") else os.path.dirname(remote_dir)

    subprocess.run(["ssh", host, f"mkdir -p {shlex.quote(remote_dir)}"], check=True)
    subprocess.run(["scp", "-p", local_path, remote_spec], check=True)
    subprocess.run(["ssh", host, f"chmod 600 {shlex.quote(remote_path)}"], check=True)
    print(f"✅ 已同步 ToBid OAuth 憑證至遠端：{remote_spec}")
    if restart_service:
        subprocess.run(
            ["ssh", host, f"cd {shlex.quote(repo_dir)} && docker compose restart {shlex.quote(restart_service)}"],
            check=True,
        )
        print(f"✅ 已重啟遠端服務：{restart_service}")


def main() -> int:
    parser = argparse.ArgumentParser(description="ToBid MCP OAuth 2.0 PKCE 登入與遠端同步工具")
    parser.add_argument("--timeout", type=int, default=180, help="等待瀏覽器授權秒數（預設 180）")
    parser.add_argument("--no-browser", action="store_true", help="不自動開啟瀏覽器，僅印出授權網址")
    parser.add_argument("--output", help="自訂輸出 token JSON 路徑（預設 cache/tender/oauth_token.json）")
    parser.add_argument("--upload-remote", help="完成後上傳至遠端（例如 10.9.0.99:/home/david/stock-underdog-ml/cache/tender/oauth_token.json）")
    parser.add_argument("--restart-service", default="", help="上傳後重啟之 docker compose service 名稱（例如 stock-ml-api）")
    args = parser.parse_args()

    try:
        saved_path = run_local_oauth_flow(
            timeout_seconds=args.timeout,
            open_browser=not args.no_browser,
            output_path=args.output,
        )
        if args.upload_remote:
            upload_to_remote(saved_path, args.upload_remote, restart_service=args.restart_service or None)
        return 0
    except Exception as e:
        print(f"❌ 失敗: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
