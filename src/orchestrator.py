import anthropic
import json

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

SYSTEM_CONTEXT = """You are a specification parser for an analog equalizer
design tool. The tool tunes a CTLE (Continuous-Time Linear Equalizer) for a
PCIe Gen2 receiver. Valid parameters:
- target_peaking_db: high-frequency peaking boost, must be between 3.0 and 11.0 dB
- noise_limit_mvrms: maximum acceptable input-referred noise, default 1.5 mVrms if not mentioned

Users describe what they want in plain language (e.g. "strong boost, don't
worry about noise" or "moderate equalization, keep it quiet"). Map qualitative
language to reasonable values within the valid ranges above.
"""


def parse_spec_request(user_text):
    """
    Converts a natural-language design request into the structured target
    spec dict EqualizerEnv expects. Falls back to a safe default if the
    model's output isn't valid JSON, so a single malformed response can't
    crash a live demo.
    """
    prompt = f"""{SYSTEM_CONTEXT}

Return ONLY a JSON object with keys target_peaking_db and noise_limit_mvrms.
No other text, no markdown formatting, just the raw JSON object.

Request: "{user_text}"
"""
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        spec = json.loads(text)
        spec['target_peaking_db'] = max(3.0, min(11.0, float(spec.get('target_peaking_db', 8.0))))
        spec['noise_limit_mvrms'] = float(spec.get('noise_limit_mvrms', 1.5))
        return spec
    except (json.JSONDecodeError, ValueError, TypeError):
        return {'target_peaking_db': 8.0, 'noise_limit_mvrms': 1.5}  # safe fallback
    except Exception as e:
        print(f"API call failed: {e}")
        return {'target_peaking_db': 8.0, 'noise_limit_mvrms': 1.5}  # safe fallback


if __name__ == '__main__':
    print(parse_spec_request("I need strong high-frequency boost, noise isn't a big concern"))
    print(parse_spec_request("moderate boost around 6dB, keep it quiet"))
