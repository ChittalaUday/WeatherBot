# WeatherBot Project Rules

## ML Model Architecture & Training Rules

When working on WeatherBot ML modeling, dataset curation, or training pipelines, enforce the rules defined in `MODEL_RULES.md`:

1. **Strict Architectural Scope**:
   - Focus solely on predicting: `weather_intent`, `action`, `LOCATION` entity spans, and `TIME` entity spans.
   - Do not perform location coordinate resolution (Solr) or time normalization (Time Parser) in the ML model.

2. **Allowed Intent Labels (14)**:
   - `CURRENT_CONDITIONS`, `FORECAST`, `TEMPERATURE`, `TEMPERATURE_MIN`, `TEMPERATURE_MAX`, `RAIN`, `HUMIDITY`, `DEW_POINT`, `WIND_SPEED`, `WIND_DIRECTION`, `SUNSHINE`, `CLOUD_COVER`, `SOIL_MOISTURE`, `SOIL_TEMPERATURE`.
   - Never use `CURRENT_WEATHER` or temporal/spatial names as intent labels.

3. **Allowed Actions (3)**:
   - `GET`, `COMPARE`, `ALERT`.

4. **Entities**:
   - `LOCATION`: Extract raw location text tokens as list of strings.
   - `TIME`: Extract raw time text tokens as list of strings.

5. **JSON Model Output Schema**:
   ```json
   {
     "weather_intent": "TEMPERATURE",
     "action": "GET",
     "entities": {
       "location": ["Hyderabad"],
       "time": ["tomorrow"]
     }
   }
   ```
