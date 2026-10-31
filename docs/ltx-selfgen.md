# LTX self-generation: long takes split into sentences

Idea: generate multi-sentence takes with the base model, cut them at sentence boundaries
(`queen.longform`), and use sentence *i* of a take as the reference for sentence *j* of the same take.
One take = one voice and one room, so every pair is a clean same-speaker pair with different words, and
the number of voices is unlimited (LTX-2 Community License: training on our own outputs is fine).

## Splitting (`queen.longform`)

Cuts come from the script, not ASR punctuation: Whisper word timestamps (`Evaluator.words`) are aligned
to the script's words after Whisper's English normalizer (`difflib`), and each boundary goes halfway
through the pause between one sentence's last word and the next one's first. A take whose boundary words
are not recognized is dropped (`sentence_cuts` → `None`). Splitting on Whisper's own sentence-final
punctuation failed on 5 of 24 pilot takes (a deep, slow male voice got no periods at all); script
alignment split 96 / 96 sentences, every segment at WER 0.00.

## Pilots (2026-10-30, base model, no reference, seed fixed per voice)

**Pilot 1** — one 4-sentence text (calm → worried → angry), 19 s fixed length, three prompt styles:
arc described in prose, an lnvox cue before each sentence, and a steady control. **Pilot 2** — 4 voice
descriptors × 3 arcs (calm→angry `rise`, angry→calm `fall`, joy→sad `turn`), each with a neutral control
take of the same text and seed; length from the duration head (8–14 s, ~3.1–3.6 words/s).

| Measure | Result |
|---|---|
| Text | WER 0.00 on all 27 takes and all 108 segments |
| Voice within a take (WavLM x-vector cosine between segments) | mean 0.940, min 0.754 |
| Same descriptor + seed, different take | mean 0.810, min 0.255 → **not** the same voice |
| Different descriptors | mean 0.622 (but max 0.979: WavLM-base-plus-sv separates voices poorly) |
| Mid-take emotion shifts (per-sentence F0 / loudness / rate vs. the control) | only high-arousal contrasts land, and not for every voice: anger on the last sentence 3 / 4 voices (+90–120 Hz F0, +1 word/s), anger on the first sentences 2 / 4, "sighing" → −4 dB 4 / 4; joy → sadness 0 / 4, "worried" 0 / 4 |
| Prose arc vs. per-sentence cues (pilot 1) | same behaviour |

The 4-class emotion classifier `superb/hubert-large-superb-er` is unusable here: it is right on IEMOCAP
(its own domain) but calls 21 / 25 RAVDESS neutral clips and every LTX segment "angry". Emotion QC needs a
model that transfers (e.g. emotion2vec+, or a dimensional arousal/valence model).

Denoise cost with the DiT resident: ~5 s per take regardless of 8–19 s length → 2.3–3.4 h of audio per
GPU-hour on the g7 (plus one Gemma prompt per take and per training segment, ~0.37 s each).

## Recipe

- **One emotion per take.** Per-sentence cues are not reliably performed, so a segment's label must be
  the take's single cue (the base model's own prompt → audio mapping, so training on it does not teach the
  model to ignore cues). Multi-sentence DeepDialogue turns with their director cue are ready-made takes.
- **Pairs only within a take.** Seeds and descriptors do not pin a voice across takes.
- **Emotion contrast comes from elsewhere:** real data (EmoV-DB, IEMOCAP: cross-emotion pairs) and
  DramaBox renders. Optional extra: calm → angry `rise` takes, kept only where QC confirms the final
  sentence moved, give same-voice cross-emotion pairs.
- **Later, bootstrap:** once a fine-tune follows text with a reference, condition new emotional takes on a
  neutral segment of an earlier take to get cross-take, cross-emotion pairs for the same voice.
