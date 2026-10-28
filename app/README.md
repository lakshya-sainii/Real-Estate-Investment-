# Estate IQ Streamlit App

Premium obsidian-and-mint interface with the original Talina/Manrope typography, a high-end architectural hero visual, interactive brand mark, and seven pages.

| Page | What it does |
|---|---|
| Home | Editorial landing page, premium skyline visual, model summary, market snapshot |
| Analyse Property | Verdict, confidence, five-year estimate, price journey, criteria and city-median comparison |
| Market Atlas | Non-geographic city-opportunity universe, signature cards, two-city radar comparison |
| Insights | Eight analytics views: overview, custom opportunity index, growth, cities, property mix, livability Sankey, correlations and CSV export |
| AI Advisor | Rule-based advisor using the project data and models; no API key needed |
| Performance | Held-out metrics, confusion matrix, feature importance and MLflow notes |
| About | Method, assumptions and limitations |

## Run from the project root

```bash
python -m pip install -r app/requirements.txt
streamlit run app/app.py
```

The app is self-contained. Models, lookups, metrics and the compact listings dataset are in `app/artifacts/`.

## Important compatibility fix

The app no longer deserialises XGBoost. Its portable scikit-learn classifier is distilled from the same five-criterion investment rule, which permanently avoids the `XGBoostError: input stream corrupted` runtime failure. The original XGBoost research model and metrics remain in the complete project, outside the app inference path.

## Notes

- Google Fonts needs internet access; the embedded Talina font and hero image work locally.
- Market Atlas uses no map tiles, three.js, WebGL or external JavaScript.
- The AI Advisor is not a generative model. It answers only from this project's data and models.
- Growth rates are assumptions derived from project rules, not observed market history. This is not financial advice.
- Talina DEMO by Pandeka Foundry is free for personal use only. Obtain a licence before public or commercial deployment.
