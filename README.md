# Top 5 Shorts: video stage

`make_video.py` turns an approved script (after Gate 2) into a checked 1080×1920 Short, following the Video Stage Build Guide. Each step saves its output into the job folder and skips itself on a re-run.

## Run it

```bash
pip install -r requirements.txt        # plus FFmpeg on your PATH
python setup_assets.py                 # fonts + free local voice model (~350 MB)
cp .env.example .env                   # add Pexels / Pixabay / ElevenLabs keys
python make_video.py jobs/2026-10-09-animal-superpowers
```

Then watch `jobs/<id>/final.mp4` once with sound (Gate 3). To replace a shot, pass its id from `timing.json`:

```bash
python make_video.py jobs/2026-10-09-animal-superpowers --swap rank3.2 rank1.0
```

Only those shots are fetched again and the video is re-rendered; the voice is reused.

## What each step makes (in `jobs/<id>/`)

| Step | Output | Notes |
| --- | --- | --- |
| 0. Script lint | printed warnings, `report.json` | The plan's word-count, hook, tease and banned-phrase checks |
| 1. Voice | `voice_tts.*`, `words.json`, `voice_norm.wav` | ElevenLabs when its keys are set, otherwise the free local Kokoro voice. Drop your own `voice.mp3` in and Whisper times it instead |
| 2. Timing map | `timing.json` | Sections, 4–6 shots per entry; hook uses #1's terms, outro reuses the hook shot so the Short loops |
| 3–4. Clips, credits | `shots.json`, `clips/`, `credits.txt`, `../used_clips.json` | Pexels → Pixabay → Wikimedia Commons image → graphic card. Skips clips used in the last 30 Shorts |
| 5. Shots | `background.mp4` | Every shot cut to its exact frame count; stills get a slow zoom |
| 6–7. Render | `final.mp4` | Hook text on frame 1, rank badges, word-by-word captions, music 18 dB under the voice, loudness −14 LUFS |
| 8. Checks | `report.json` | Length, format, loudness, black/frozen frames, safe zone, credits |
| 9. Upload text | `description.txt` | Title, AI-disclosure flag, sources, footage credits, hashtags |

## Visual fallbacks

If no stock clip fits a shot, it becomes a graphic card in the section's colours with the animal's name. `report.json` warns with the count. A Short with cards is a draft, not ready to publish: rerun with Pexels/Pixabay keys (and network access to those APIs) to get real footage.

## Next steps

1. **Gate 3 for this Short:** run with Pexels and Pixabay keys so every shot has real footage, add 10–20 YouTube Audio Library tracks under `assets/music/curious/`, watch once and swap weak shots.
2. **Upload:** paste `description.txt` into YouTube Studio by hand until a Publish script exists (YouTube Data API `videos.insert`, with `status.publishAt` and `status.containsSyntheticMedia`).
3. **Voice decision:** your own recording (`voice.mp3`) is the strongest authorship signal; otherwise pick one voice (ElevenLabs or Kokoro) and keep it for every Short.
4. **Tracker and scheduling:** wire this script to the content tracker so "Script approved" starts it, and report pass/fail back to the tracker.
