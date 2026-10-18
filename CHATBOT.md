# AI assistant implementation notes

Verified against official OpenAI documentation on 18 October 2026. No API credential was supplied, so no authenticated model call was made. Offline tests use mocked HTTP responses and do not spend API credit.

## Recommended provider and model

OpenAI's Responses API is a practical choice because it supports the requested text conversation and streaming with one server-side endpoint. The helper uses Python's standard HTTP and JSON libraries, so adding the chatbot does not require upgrading the machine-learning packages or installing another SDK.

| Setting | Recommended default | Lower-cost alternative |
|---|---|---|
| Provider | OpenAI | OpenAI |
| `OPENAI_MODEL` | `gpt-5.4-mini` | `gpt-6-luna` |
| Standard input, per 1 million tokens | USD 0.75 | USD 0.10 |
| Standard output, per 1 million tokens | USD 4.50 | USD 0.50 |
| Cached input, per 1 million tokens | USD 0.075 | USD 0.01 |
| Integration | Responses API, streaming | Same helper and API key |

Model pages opened and read:

- https://developers.openai.com/api/docs/models/gpt-5.4-mini
- https://developers.openai.com/api/docs/models/gpt-6-luna

The recommendation is a practical default rather than a measured quality comparison for this particular farming application. Test sample questions on the cheaper model before switching. Availability depends on the API account's access and billing; the published free tier is not supported for these models.

For illustration, 1,000 input tokens plus 300 output tokens on the default model cost approximately USD 0.00210 at these standard rates; on the alternative, USD 0.00025. This excludes taxes, currency conversion, any cache-write pricing and additional usage. Input includes system instructions, recent conversation history and any opted-in estimate summary, so a question's visible length alone is not its billed input. Actual charges depend on metered token usage and current pricing.

The `gpt-5-mini` and `gpt-5.4-nano` pages were marked **Deprecated**, so they were not selected despite their low prices. The supported `gpt-5.4-mini` default is configurable without changing the trained model.

## Request and streaming behaviour

- Endpoint: `POST https://api.openai.com/v1/responses`.
- Sends `stream: true`, `store: false`, `tools: []`, `max_output_tokens: 1600`.
- Uses `reasoning: {effort: "none"}` on the documented default and alternatives to reserve the small output budget for a visible reply. This setting is supported on these models.
- Reads `response.output_text.delta` and `response.refusal.delta`. Requires `response.completed` to classify a reply as finished. Handles `response.incomplete`, `response.failed`, `error`, malformed frames, disconnects and empty output.
- No automatic retries, so a failed request does not silently trigger additional billable calls.
- OpenAI recommends SDKs for convenience; this application deliberately uses the documented HTTP/SSE interface to preserve dependency compatibility.

Sources:

- https://developers.openai.com/api/docs/guides/streaming-responses
- https://developers.openai.com/api/reference/resources/responses/streaming-events
- https://developers.openai.com/api/reference/python/resources/responses/methods/create

## Configuration, privacy and honest scope

`load_config(st.secrets)` reads `OPENAI_API_KEY` and optional `OPENAI_MODEL` from root-level Streamlit secrets or environment variables. Environment values take priority. Placeholder keys count as unconfigured. The API key is excluded from dataclass representations and never included in displayed error messages. The fixed destination is the OpenAI API.

Configuration means a key was supplied; it does not prove that the credential, model access or billing works. A real successful call remains an owner verification step.

The UI must obtain explicit user opt-in before it passes `prediction_context`. The helper accepts only a single summary with whitelisted input/output fields and short notes. It rejects batch records, unexpected identifier fields, nested collections, nonfinite values and oversized context. Ordinary typed chat is still sent to the AI provider, so do not put sensitive records into a question.

`store: false` disables storage of the generated response for later API retrieval. It is **not** a promise of zero retention: default abuse monitoring logs may contain content and remain for up to 30 days, with policy exceptions. The app keeps the visible conversation in its current Streamlit session rather than creating a provider-hosted conversation object.

There is no connected weather, mandi price, government scheme or search service. The model is instructed to state that limitation, distinguish general information from current facts and never fabricate source citations. It is also instructed never to manufacture model predictions, accuracy metrics, thresholds or ranges. Such model instructions reduce errors but cannot guarantee every generated answer is correct.

Sources:

- API key quickstart: https://developers.openai.com/api/docs/quickstart
- API key dashboard: https://platform.openai.com/api-keys
- Data controls: https://developers.openai.com/api/docs/guides/your-data

## UI integration example

```python
from assistant import AssistantError, configured, load_config, stream_reply

config = load_config(st.secrets)
# Messages must end with the current user's question.
# Context must be None unless a user explicitly shares their single estimate.
try:
    answer = st.write_stream(stream_reply(messages, config, prediction_context))
except AssistantError as exc:
    st.warning(str(exc))
    # If exc.partial, label the displayed text incomplete; do not save it as a
    # completed assistant reply in the future conversation sent to the API.
```

The UI must collect the yielded text itself if it needs to retain partial text when `st.write_stream` raises; the generator does not mutate chat history. It has no model-loading or Streamlit imports and cannot block the estimator at import time.

Run offline verification from the project directory:

```bash
python -m unittest discover -s tests -p 'test_assistant.py' -v
```

During development the same tests can run beside the helper with `python -m unittest -v test_assistant.py`. The unit-test-only key is never sent to a live service.
