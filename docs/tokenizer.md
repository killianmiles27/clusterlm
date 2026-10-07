# Father tokenizer and chat template

Token IDs, text and the vocabulary exist only on Father (privacy rule: no token IDs, text, prompts or logits on any
Node-facing protocol). Father builds each tier's real tokenizer from the **metadata** of the tier model's first GGUF
shard (the shard listed first in the model directory's manifest). Tensors are never read.

## What is supported

`father::GgufBpeTokenizer` (`orchestrator/father-service`) re-implements llama.cpp's byte-level BPE for
`tokenizer.ggml.model == "gpt2"`:

| `tokenizer.ggml.pre` | Pre-tokenizer |
|---|---|
| `qwen2` | `'[sS]\|'[tT]\|'[rR][eE]\|'[vV][eE]\|'[mM]\|'[lL][lL]\|'[dD]\|[^\r\n\p{L}\p{N}]?\p{L}+\|\p{N}\| ?[^\s\p{L}\p{N}]+[\r\n]*\|\s*[\r\n]+\|\s+(?!\S)\|\s+` |
| `qwen35` | the same with `\p{M}` added to the letter class of alternatives 2 and 4 |

Any other `pre`, a missing `pre`, or another tokenizer model is refused with `kUnimplemented`; a tokenizer is never
guessed. The pattern is matched by hand (leftmost-first alternatives with the backtracking the pattern needs); there
is no `std::regex`. Letter/mark/number classes come from a compact generated table
(`orchestrator/father-service/src/unicode_tables.cpp`, produced by `scripts/gen_unicode_tables.py` from Python's
`unicodedata`, version recorded in the file header); whitespace is the fixed Unicode `White_Space` set.

Pipeline: special-token partition (control, user-defined, unknown tokens, longest text first; control and unknown are
only matched with `parse_special`) -> pre-tokenizer -> GPT-2 byte-to-unicode map -> merge ranks (lowest rank first,
ties leftmost) -> token ids. Decode maps token text back through the byte map; control tokens render as nothing.

Differences from llama.cpp, all deliberate: invalid UTF-8 is tokenized as single byte units (llama.cpp throws), and a
vocabulary lacking any of the 256 byte-level base tokens is rejected at construction (llama.cpp silently drops such
bytes). Together these make `decode(encode(x)) == x` hold for **every** byte string. Special-token ties of equal
length are broken by id. `encode()` never parses control tokens, so user text cannot inject `<|im_start|>`.

## Chat template

There is no Jinja engine. `ChatMlTemplate` renders Qwen ChatML
(`<|im_start|>role\ncontent<|im_end|>\n ... <|im_start|>assistant\n`) when `tokenizer.chat_template` contains
`<|im_start|>`; any other or missing template makes the tokenizer (and so the tier) fail with `kUnimplemented` and the
reason. Read from the template text by substring only: the Qwen2-style default system turn
("You are a helpful assistant"), the Qwen3 thinking toggle (`enable_thinking=false` appends the empty
`<think>\n\n</think>\n\n` block, via `GgufBpeOptions::chat.enable_thinking`), and the Qwen3.5-style open `<think>\n`
after the assistant opener when thinking is enabled. Template markers become token ids; message content is plain text
tokenized without control tokens. Not modelled: stripping reasoning from earlier assistant turns, tool calls, images.

Stop tokens: `Tokenizer::stop_token_ids()` returns eos, eot, `<|im_end|>` and `<|endoftext|>` (those present); the
Father service passes them as `GenerationRequest::stop_tokens`.

## Wiring

`TierTokenizerProvider` caches one tokenizer per tier. It feeds `ServiceDeps::tokenizer_for_tier` (prompt, decode,
stop tokens) and `ProductionOptions::tokenizers` (readiness: `ReadinessInputs::tokenizer_problem` makes the tier
Unavailable with the reason; there is no byte-tokenizer fallback). The fixture path (`--dev-fixture-model`) keeps
`FixtureByteTokenizer`. A lower tier whose tokenizer differs from the failing tier's is not a downgrade target for an
in-flight answer. The Fast tier uses `GgufBpeTokenizer` too: llama.cpp's tokenizer is not exposed by
`runtime/backends/llama`.

## Verification

`tests/father_service/test_vocab_llamacpp.cpp` runs every case of llama.cpp's `ggml-vocab-qwen2` and
`ggml-vocab-qwen35` (`.inp`/`.out`, add_special=false, parse_special=false, as in `test-tokenizer-0.cpp`) and requires
exact id equality plus an exact decode round trip. Result: qwen2 46/46, qwen35 50/50. The files come from the pinned
llama.cpp checkout (`python3 scripts/fetch_upstream.py llama.cpp`); the test is registered (ctest label `vocab`) only
when they exist (`-DCLUSTERLM_LLAMACPP_VOCAB_DIR=...` overrides the location) and CI fetches them in the gcc and
llama jobs. Synthetic-vocabulary unit tests (`test_gguf_bpe.cpp`) and the `fuzz_gguf_bpe_tokenizer` target (round trip
on arbitrary bytes) run everywhere. These vocabularies are 150k-token Qwen2 files; Qwen3.8 vocabularies are checked
the same way once their GGUFs are available (not measured here).
