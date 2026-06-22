/**
 * The Loom result envelope — the TS mirror of the Python dual-driver contract.
 *
 * LOCKED. These names match `loom/types.py:VerbResult.to_dict()` verbatim (12
 * top-level keys, stable order). `python -m loom <verb> --json` prints exactly
 * this object as its trailing stdout line; the bridge (`parseLoomJson`) parses it
 * into a `VerbResult`. Do NOT add/rename a key without threading it through the
 * Python `to_dict` first — the byte-identical dual-driver envelope IS the contract.
 *
 * Source of record: /Users/anub/Work/transaction-foundation-model/Loom/loom/types.py
 */

/** Top-level call status (`loom/types.py:Status`). `PLAN` carries a confirm_token
 *  for the agent's second call; the `REFUSED_*` family are structural refusals. */
export type LoomStatus =
	| "OK"
	| "PLAN"
	| "REFUSED_NO_METRIC"
	| "REFUSED_NO_BASELINE"
	| "REFUSED_NO_GPU_TARGET"
	| "REFUSED_AGENT_CANNOT_LAUNCH"
	| "REFUSED_NONINTERACTIVE_LAUNCH"
	| "REFUSED_SPEND_CAP"
	| "REFUSED_STALE"
	| "REFUSED_CONTRACT"
	| "FAIL";

/** The machine-checkable outcome of a verb (`loom/types.py:Verdict`). */
export type LoomVerdict = "PASS" | "REVIEW" | "FAIL" | "INCOMPLETE";

/** A property of the verb, not a flag (`loom/types.py:Tier`). Gating reads it. */
export type LoomTier = "read-only" | "workspace-write" | "expensive" | "irreversible";

/** How a verb participates in the search/launch machinery (`loom/types.py:CapabilityMode`). */
export type LoomCapabilityMode = "none" | "searchable" | "launch-and-track";

/** Severity of a single contract diagnostic (`loom/types.py:Severity`). */
export type LoomSeverity = "info" | "warning" | "error";

/**
 * One named contract finding — the named-diff card, not a stack trace
 * (`loom/types.py:Diagnostic.to_dict`). Each element of `VerbResult.diagnostics`.
 * The contract-diff widget renders `data` (ids/deltas) via `renderDiff`, framed
 * by `contract` / `severity` / `fix`.
 */
export interface Diagnostic {
	/** "C1" | "C2" | "C3" | "C6" | "EDA" | ... */
	contract: string;
	severity: LoomSeverity;
	message: string;
	/** the offered one-line fix; may be null/absent. */
	fix?: string | null;
	/** structured detail (ids, deltas) — drives the named-diff body. */
	data: Record<string, unknown>;
}

/**
 * A *derived* cost plan (`loom/types.py:CostPlan.to_dict`). PLACEHOLDER in the
 * Phase-0 slice: the fields the gating model will use are present and carried on
 * the envelope, but the three CPU verbs leave them null/zero. `derived`
 * distinguishes a computed estimate from a label.
 */
export interface CostPlan {
	derived: boolean;
	usd?: number | null;
	/** "LOW" | "MEDIUM" | "HIGH" */
	confidence?: string | null;
	tokens?: number | null;
	params?: number | null;
	seq_len?: number | null;
	gpu_target?: string | null;
	/** the binding envelope a human approves (§4.3); null until a launch verb fills it. */
	envelope?: Record<string, unknown> | null;
	inputs: Record<string, unknown>;
}

/**
 * The single result envelope shared by both driver faces
 * (`loom/types.py:VerbResult.to_dict`). LOCKED shape: 12 keys, stable order.
 */
export interface VerbResult {
	verb: string;
	status: LoomStatus;
	verdict: LoomVerdict;
	tier: LoomTier;
	capability_mode: LoomCapabilityMode;
	summary: string;
	/** pathspec strings, e.g. ["Corpus/1"]. */
	outputs: string[];
	diagnostics: Diagnostic[];
	data: Record<string, unknown>;
	experiment?: string | null;
	cost_plan?: CostPlan | null;
	/** single-use, plan-hash-scoped, expiring token; null unless status === "PLAN". */
	confirm_token?: string | null;
}

/**
 * Loom-internal metadata carried alongside (not part of) the Anthropic tool
 * schema — the `_loom` block emitted by `loom/tools.py:tool_schema`. The
 * extension READS `disable_model_invocation` to decide whether a verb gets a
 * `tool_call` {block} gate + an omitted `promptSnippet`. It is invisible to Pi;
 * the gate hook (gate.ts) is the only real lock.
 */
export interface LoomToolMeta {
	tier: LoomTier;
	capability_mode: LoomCapabilityMode;
	disable_model_invocation: boolean;
}

/**
 * One verb's tool schema as emitted by `python -m loom verbs --json` — a JSON
 * array of these. `input_schema` is the full JSON-Schema `params` object
 * (`type:"object"` + `properties`), NOT the legacy required/optional flag lists.
 * The bridge converts `input_schema` → a TypeBox `TSchema` for Pi's registerTool.
 */
export interface VerbSchema {
	/** dotted engine name, e.g. "loom.tokenize". */
	name: string;
	description: string;
	/** JSON-Schema object: { type:"object", properties:{...}, required?:[...] }. */
	input_schema: JsonSchemaObject;
	_loom: LoomToolMeta;
}

/** The subset of JSON-Schema the engine emits for a verb's params. */
export interface JsonSchemaObject {
	type: "object";
	properties: Record<string, JsonSchemaProperty>;
	required?: string[];
	additionalProperties?: boolean;
}

/** A single JSON-Schema property in a verb's `input_schema.properties`. */
export interface JsonSchemaProperty {
	type?: "string" | "integer" | "number" | "boolean" | "array" | "object";
	description?: string;
	enum?: Array<string | number>;
	items?: JsonSchemaProperty;
	[key: string]: unknown;
}
