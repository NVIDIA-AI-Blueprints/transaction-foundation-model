# Loom CLI — structural tests (Implement C)

Model-free structural tests for the Loom-on-Pi v0.1 slice. They verify the bridge
parse, the contract-diff widget, and tool/gate registration **without** booting a
real Pi session or making any model call. The only subprocess is one CPU-only
engine verb (`loom tokenize`) in the parse test.

## Run

From `cli/` (after `npm run build`, which the tests import from `dist/`):

```sh
npm test                 # node --test --experimental-strip-types "test/**/*.test.ts"
npm run typecheck:test   # tsc -p tsconfig.test.json (type-checks the tests)
```

Or directly:

```sh
node --test --experimental-strip-types "test/**/*.test.ts"
```

Requires Node ≥ 22.19 (uses built-in `node:test` + `--experimental-strip-types`;
no `tsx`/jest). Test sources are `.ts`; they import compiled engine code from
`../dist/*.js` and sibling helpers with explicit `.ts` specifiers.

`LOOM_PYTHON` overrides the engine interpreter (defaults to the dev-box venv at
`/Users/anub/Work/transaction-foundation-model/Loom/.venv/bin/python`). If it is
absent the live parse test self-skips; all other tests still run.

## Files

- `parse.test.ts` — runs `loom tokenize --preset financial --json` once, asserts
  `parseLoomJson` yields `status OK / verdict PASS / outputs ["Corpus/1"] /
  data.vocab_size 6251`, all 12 envelope keys, and trailing-JSON robustness.
- `widget.test.ts` — `renderContractResult` on a hand-built FAIL/C1 envelope;
  asserts `.render(80)` lines surface the contract id, message, and fix; plus the
  empty-`details` fallback to text content.
- `registration.test.ts` — mock `ExtensionAPI`; `registerLoomTools` +
  `installLoomGate` register exactly `loom_tokenize/ingest/baseline` with the
  locked tool surface, populate the tier registry (Phase-0 ungated), and install a
  `tool_call` hook that is inert for ungated verbs in both `tui` and `rpc` modes.
- `helpers/mock-pi.ts` — capturing mock of Pi 0.79.0 `ExtensionAPI` (`registerTool`
  / `on`), plus synthetic `tool_call` event + `ExtensionContext` builders.
- `helpers/fixtures.ts` — envelope fixtures, a real `Theme` (full color record),
  render option/context stubs, and `stripAnsi`.
