# AI Daily Summary Pipeline

Daily automated pipeline that:

1. Identifies top AI YouTube channels
2. Retrieves latest videos
3. Extracts transcripts
4. Summarizes intelligently (deduplicated + ranked)
5. Generates podcast-quality audio
6. Publishes a private RSS feed

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python pipeline.py run --config config.yaml
```

## Notes

- YouTube discovery + video retrieval uses YouTube Data API v3 (`YOUTUBE_API_KEY`).
- Transcript extraction uses `youtube-transcript-api`.
- Summarization + ranking + TTS use OpenAI (`OPENAI_API_KEY`) via model names in `config.yaml`.
- Private RSS feed is generated as `output/private_feed.xml` and can optionally be uploaded to S3.

## Cron

Example daily cron at 7:00 AM UTC:

```cron
0 7 * * * cd /workspace/AI_Daily_Summary && /usr/bin/python3 pipeline.py run --config config.yaml >> output/pipeline.log 2>&1
```
