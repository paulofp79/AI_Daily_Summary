import argparse
import datetime as dt
import hashlib
import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi


@dataclass
class VideoItem:
    video_id: str
    title: str
    channel_id: str
    channel_title: str
    published_at: str
    description: str
    transcript: str = ""
    relevance_score: float = 0.0
    summary: str = ""


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dir(config: dict[str, Any]) -> Path:
    out = Path(config["storage"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out


def youtube_api_get(api_key: str, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"https://www.googleapis.com/youtube/v3/{endpoint}"
    params = {**params, "key": api_key}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def discover_top_channels(config: dict[str, Any], api_key: str) -> list[dict[str, Any]]:
    disc = config["discovery"]
    channel_ids: set[str] = set()

    for term in disc["search_terms"]:
        data = youtube_api_get(
            api_key,
            "search",
            {
                "part": "snippet",
                "q": term,
                "type": "channel",
                "maxResults": disc["max_channels_per_term"],
                "order": "relevance",
            },
        )
        for item in data.get("items", []):
            cid = item["snippet"].get("channelId")
            if cid:
                channel_ids.add(cid)

    if not channel_ids:
        return []

    stats = youtube_api_get(
        api_key,
        "channels",
        {
            "part": "snippet,statistics",
            "id": ",".join(sorted(channel_ids)),
            "maxResults": 50,
        },
    )

    min_subs = int(disc["min_subscribers"])
    channels = []
    for c in stats.get("items", []):
        subs = int(c.get("statistics", {}).get("subscriberCount", 0))
        if subs >= min_subs:
            channels.append(
                {
                    "channel_id": c["id"],
                    "title": c["snippet"]["title"],
                    "subscriber_count": subs,
                }
            )

    channels.sort(key=lambda x: x["subscriber_count"], reverse=True)
    return channels


def fetch_latest_videos(config: dict[str, Any], api_key: str, channels: list[dict[str, Any]]) -> list[VideoItem]:
    lookback_hours = int(config["videos"]["lookback_hours"])
    max_videos_total = int(config["videos"]["max_videos_total"])
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=lookback_hours)).isoformat()

    videos: list[VideoItem] = []
    for channel in channels:
        data = youtube_api_get(
            api_key,
            "search",
            {
                "part": "snippet",
                "channelId": channel["channel_id"],
                "type": "video",
                "order": "date",
                "publishedAfter": since,
                "maxResults": 10,
            },
        )
        for item in data.get("items", []):
            snippet = item["snippet"]
            videos.append(
                VideoItem(
                    video_id=item["id"]["videoId"],
                    title=snippet.get("title", ""),
                    channel_id=snippet.get("channelId", ""),
                    channel_title=snippet.get("channelTitle", ""),
                    published_at=snippet.get("publishedAt", ""),
                    description=snippet.get("description", ""),
                )
            )
            if len(videos) >= max_videos_total:
                return videos

    return videos


def fetch_transcript(video_id: str) -> str:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except Exception:
        return ""


def attach_transcripts(videos: list[VideoItem]) -> list[VideoItem]:
    for v in videos:
        v.transcript = fetch_transcript(v.video_id)
    return videos


def dedupe_videos(videos: list[VideoItem]) -> list[VideoItem]:
    seen = set()
    output = []
    for v in videos:
        sig = hashlib.sha256((v.title.lower() + v.channel_id).encode("utf-8")).hexdigest()
        if sig in seen:
            continue
        seen.add(sig)
        output.append(v)
    return output


def summarize_and_rank(config: dict[str, Any], client: OpenAI, videos: list[VideoItem]) -> tuple[str, list[VideoItem]]:
    payload = [
        {
            "video_id": v.video_id,
            "title": v.title,
            "channel": v.channel_title,
            "published_at": v.published_at,
            "description": v.description,
            "transcript": v.transcript[:8000],
        }
        for v in videos
    ]

    prompt = (
        "You are an expert AI news editor. Given video metadata and transcripts, "
        "produce a concise daily digest and score each video 0-1 by relevance/novelty. "
        "Return JSON with keys: digest_markdown, items[]. Each item must contain video_id, summary, relevance_score."
    )

    resp = client.chat.completions.create(
        model=config["summarization"]["model"],
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload)},
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    data = json.loads(content)

    by_id = {v.video_id: v for v in videos}
    for item in data.get("items", []):
        vid = item.get("video_id")
        if vid in by_id:
            by_id[vid].summary = item.get("summary", "")
            by_id[vid].relevance_score = float(item.get("relevance_score", 0.0))

    threshold = float(config["summarization"]["min_relevance_score"])
    keep = [v for v in videos if v.relevance_score >= threshold]
    keep.sort(key=lambda x: x.relevance_score, reverse=True)
    keep = keep[: int(config["summarization"]["max_items_in_digest"])]
    return data.get("digest_markdown", ""), keep


def make_audio_script(digest_markdown: str, ranked: list[VideoItem]) -> str:
    lines = [
        "Welcome to your AI Daily Summary.",
        "Here are today's most important AI videos and takeaways.",
        "",
    ]
    if digest_markdown:
        lines.append(digest_markdown)
        lines.append("")
    for idx, v in enumerate(ranked, 1):
        lines.append(f"{idx}. {v.title} by {v.channel_title}. {v.summary}")
    lines.append("")
    lines.append("That's all for today. See you tomorrow.")
    return "\n".join(lines)


def generate_audio(config: dict[str, Any], client: OpenAI, text: str, output_path: Path) -> Path:
    model = config["audio"]["model"]
    voice = config["audio"]["voice"]
    fmt = config["audio"]["format"]
    audio = client.audio.speech.create(model=model, voice=voice, input=text, format=fmt)
    out_file = output_path / f"digest_{dt.datetime.now().strftime('%Y%m%d')}.{fmt}"
    audio.stream_to_file(str(out_file))
    return out_file


def write_rss(config: dict[str, Any], items: list[VideoItem], audio_path: Path) -> Path:
    rss = config["rss"]
    token = rss["private_token"]
    pub_date = dt.datetime.now(dt.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    enclosure_url = f"{rss['enclosure_base_url'].rstrip('/')}/{audio_path.name}?token={token}"

    item_body = "\n".join(
        [
            f"<li><a href='https://www.youtube.com/watch?v={html.escape(v.video_id)}'>{html.escape(v.title)}</a>"
            f" ({html.escape(v.channel_title)}) - score {v.relevance_score:.2f}<br/>{html.escape(v.summary)}</li>"
            for v in items
        ]
    )

    xml = f"""<?xml version='1.0' encoding='UTF-8'?>
<rss version='2.0'>
  <channel>
    <title>{html.escape(rss['title'])}</title>
    <link>{html.escape(rss['site_url'])}</link>
    <description>{html.escape(rss['description'])}</description>
    <item>
      <title>AI Daily Summary - {dt.datetime.now().strftime('%Y-%m-%d')}</title>
      <description><![CDATA[<ul>{item_body}</ul>]]></description>
      <pubDate>{pub_date}</pubDate>
      <enclosure url='{html.escape(enclosure_url)}' type='audio/mpeg' />
      <guid>{html.escape(enclosure_url)}</guid>
    </item>
  </channel>
</rss>
"""

    out_file = Path(rss["output_file"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(xml, encoding="utf-8")
    return out_file


def maybe_upload_s3(config: dict[str, Any], file_path: Path) -> None:
    bucket = (config.get("storage", {}) or {}).get("s3_bucket", "")
    if not bucket:
        return
    prefix = config["storage"].get("s3_prefix", "")
    key = f"{prefix.rstrip('/')}/{file_path.name}" if prefix else file_path.name

    import boto3

    s3 = boto3.client("s3")
    s3.upload_file(str(file_path), bucket, key)


def run(config_path: str) -> None:
    load_dotenv()
    config = load_config(config_path)
    out_dir = ensure_output_dir(config)

    youtube_api_key = os.environ.get("YOUTUBE_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not youtube_api_key:
        raise RuntimeError("YOUTUBE_API_KEY is required")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    client = OpenAI(api_key=openai_key)

    channels = discover_top_channels(config, youtube_api_key)
    videos = fetch_latest_videos(config, youtube_api_key, channels)
    videos = dedupe_videos(attach_transcripts(videos))
    digest_markdown, ranked = summarize_and_rank(config, client, videos)

    script = make_audio_script(digest_markdown, ranked)
    script_path = out_dir / f"script_{dt.datetime.now().strftime('%Y%m%d')}.txt"
    script_path.write_text(script, encoding="utf-8")

    audio_path = generate_audio(config, client, script, out_dir)
    rss_path = write_rss(config, ranked, audio_path)

    maybe_upload_s3(config, audio_path)
    maybe_upload_s3(config, rss_path)

    print(f"Done. Channels={len(channels)} videos={len(videos)} ranked={len(ranked)}")
    print(f"Audio: {audio_path}")
    print(f"RSS: {rss_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Daily Summary pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_cmd = sub.add_parser("run", help="Run daily pipeline")
    run_cmd.add_argument("--config", default="config.yaml")

    args = parser.parse_args()
    if args.cmd == "run":
        run(args.config)


if __name__ == "__main__":
    main()
