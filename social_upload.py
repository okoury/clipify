"""
social_upload.py — modular social media upload for Snipflow.

Environment variables:
  YOUTUBE_CLIENT_SECRETS_FILE   path to OAuth 2.0 client_secret.json
  YOUTUBE_CREDENTIALS_FILE      where to cache the OAuth token (default: youtube_token.pkl)
  TIKTOK_CLIENT_KEY             (future)
  TIKTOK_CLIENT_SECRET          (future)
  INSTAGRAM_ACCESS_TOKEN        (future)
  INSTAGRAM_USER_ID             (future)
"""

import os


def upload_clip(platform: str, file_path: str, title: str,
                description: str = "") -> dict:
    """Dispatch upload to the requested platform."""
    handlers = {
        "youtube":   _upload_youtube,
        "tiktok":    _upload_tiktok_stub,
        "instagram": _upload_instagram_stub,
    }
    handler = handlers.get(platform)
    if not handler:
        return {"ok": False, "error": f"Unknown platform '{platform}'"}
    return handler(file_path, title, description)


# ── YouTube Shorts ─────────────────────────────────────────────────────────────

def _upload_youtube(file_path: str, title: str, description: str) -> dict:
    """
    Upload a clip to YouTube using the Data API v3.
    The video is uploaded as a YouTube Short (vertical, ≤ 60 s recommended).

    Requires:
      pip install google-api-python-client google-auth-oauthlib
    """
    secrets_file = os.environ.get("YOUTUBE_CLIENT_SECRETS_FILE", "")
    token_file   = os.environ.get("YOUTUBE_CREDENTIALS_FILE", "youtube_token.pkl")

    if not secrets_file or not os.path.exists(secrets_file):
        return {
            "ok":    False,
            "error": "Set YOUTUBE_CLIENT_SECRETS_FILE to your OAuth client_secret.json path.",
        }

    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        import pickle
    except ImportError:
        return {
            "ok":    False,
            "error": "Run: pip install google-api-python-client google-auth-oauthlib",
        }

    SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
    creds  = None

    if os.path.exists(token_file):
        with open(token_file, "rb") as fh:
            creds = pickle.load(fh)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow  = InstalledAppFlow.from_client_secrets_file(secrets_file, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_file, "wb") as fh:
            pickle.dump(creds, fh)

    try:
        youtube = build("youtube", "v3", credentials=creds)
        body = {
            "snippet": {
                "title":       title[:100],
                "description": description[:5000] if description else title,
                "tags":        ["shorts", "snipflow"],
                "categoryId":  "22",   # People & Blogs
            },
            "status": {
                "privacyStatus":           "public",
                "selfDeclaredMadeForKids": False,
            },
        }
        media = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True,
                                chunksize=1024 * 1024)
        request  = youtube.videos().insert(part="snippet,status", body=body,
                                           media_body=media)
        response = None
        while response is None:
            _, response = request.next_chunk()

        video_id = response["id"]
        return {
            "ok":       True,
            "video_id": video_id,
            "url":      f"https://youtube.com/shorts/{video_id}",
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ── TikTok (stub) ──────────────────────────────────────────────────────────────

def _upload_tiktok_stub(file_path: str, title: str, description: str) -> dict:
    """
    TODO: TikTok Content Posting API.

    Limitations:
    - Requires TikTok for Developers account + app review/approval.
    - Direct video upload via Content Posting API requires creator or business
      account verification; personal accounts are not eligible.
    - Auth is OAuth 2.0 with PKCE; each user must individually authorize the app.
    - Raw file upload (not a URL) requires the chunked upload endpoint.
    - Docs: https://developers.tiktok.com/doc/content-posting-api-get-started/

    Required env vars (when implemented):
      TIKTOK_CLIENT_KEY
      TIKTOK_CLIENT_SECRET
    """
    return {
        "ok":     False,
        "error":  "TikTok upload is not yet implemented.",
        "detail": (
            "TikTok's Content Posting API requires creator/business account verification "
            "and app approval. Set TIKTOK_CLIENT_KEY + TIKTOK_CLIENT_SECRET once approved."
        ),
    }


# ── Instagram Reels (stub) ─────────────────────────────────────────────────────

def _upload_instagram_stub(file_path: str, title: str, description: str) -> dict:
    """
    TODO: Instagram Reels via Graph API.

    Limitations:
    - Requires a Facebook Developer app linked to an Instagram Business or Creator account.
    - Upload is a two-step process: POST to create a container (with a publicly
      accessible video URL), then POST to publish — raw local files are not accepted.
    - The video must be hosted at a publicly reachable URL before upload.
    - Docs: https://developers.facebook.com/docs/instagram-api/guides/reels

    Required env vars (when implemented):
      INSTAGRAM_ACCESS_TOKEN
      INSTAGRAM_USER_ID
    """
    return {
        "ok":     False,
        "error":  "Instagram Reels upload is not yet implemented.",
        "detail": (
            "Instagram Graph API requires a publicly accessible video URL and a "
            "Business/Creator account. Set INSTAGRAM_ACCESS_TOKEN + INSTAGRAM_USER_ID."
        ),
    }
