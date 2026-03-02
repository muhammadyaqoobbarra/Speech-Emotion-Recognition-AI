# Contributing to EmoSense

## Areas to Contribute

- 🎙️ **New datasets** — Add support for MSP-IMPROV, IEMOCAP, or others
- 🧠 **Model improvements** — Transformers (wav2vec2, HuBERT fine-tuning)
- 🌍 **Multi-language** — Extend beyond English to cross-lingual emotion recognition
- 📱 **Mobile** — React Native app with on-device inference
- 🔄 **Streaming** — WebSocket real-time streaming analysis
- 📊 **Dashboard** — Session history, emotion timeline tracking

## Setup

```bash
git clone https://github.com/yourusername/speech-emotion.git
cd speech-emotion

# Backend
cd backend && pip install -r requirements.txt && python main.py

# Frontend
cd frontend && npm install && npm start
```

## Tests

```bash
cd backend
pytest ../tests/ -v
```

## PR Guidelines

1. Fork and create a feature branch
2. Add/update tests
3. Ensure `pytest` passes and frontend builds
4. Open a PR with a clear description

## Code Style

- Python: PEP8, max 110 chars
- JavaScript: ESLint react-app config

## License

MIT — contributions are licensed under the same terms.
