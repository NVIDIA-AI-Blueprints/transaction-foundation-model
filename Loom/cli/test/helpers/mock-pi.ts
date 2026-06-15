/**
 * A model-free mock of Pi's `ExtensionAPI` for structural tests.
 *
 * It captures every `registerTool(...)` and `on(event, handler)` call the Loom
 * extension makes, so a test can assert WHAT was wired without ever booting a
 * real Pi session, spawning the `pi` binary, or making a model call. The mock
 * implements only the surface the Loom extension touches in the factory phase
 * (`registerTool`, `on`); every other `ExtensionAPI` method is a throwing stub so
 * an accidental action-method call in the factory is caught loudly rather than
 * silently no-oping.
 *
 * Typed against Pi 0.79.0's real `ExtensionAPI`/`ToolDefinition`/event types so
 * the captured shapes match what Pi would hand the extension at runtime.
 */
import type {
	ExtensionAPI,
	ExtensionContext,
	ToolCallEvent,
	ToolCallEventResult,
	ToolDefinition,
} from "@earendil-works/pi-coding-agent";

/** A captured `on(event, handler)` registration. */
export interface CapturedHandler {
	event: string;
	// The handler is invoked by tests with a synthetic event + ctx.
	handler: (event: unknown, ctx: unknown) => unknown;
}

/** What the mock records for inspection by tests. */
export interface MockPi {
	/** The live `ExtensionAPI` to pass into the extension / registerLoomTools. */
	api: ExtensionAPI;
	/** Every tool registered via `pi.registerTool(...)`, in registration order. */
	tools: ToolDefinition<any, any, any>[];
	/** Every `pi.on(event, handler)` registration, in order. */
	handlers: CapturedHandler[];
	/** The registered tool names, in order. */
	toolNames(): string[];
	/** The first `tool_call` handler registered, or undefined. */
	toolCallHandler(): CapturedHandler | undefined;
}

/**
 * Build a mock `ExtensionAPI` that records registrations. Methods the Loom
 * factory never calls throw, so a test surfaces any unexpected action-phase call.
 */
export function makeMockPi(): MockPi {
	const tools: ToolDefinition<any, any, any>[] = [];
	const handlers: CapturedHandler[] = [];

	const unsupported = (name: string) => () => {
		throw new Error(`MockPi: ${name}() called — not supported in a structural test (no model/session).`);
	};

	const api = {
		registerTool: ((tool: ToolDefinition<any, any, any>) => {
			tools.push(tool);
		}) as ExtensionAPI["registerTool"],
		on: ((event: string, handler: (e: unknown, ctx: unknown) => unknown) => {
			handlers.push({ event, handler });
		}) as unknown as ExtensionAPI["on"],
		// Surface accidental action/registration calls the factory shouldn't make.
		registerCommand: unsupported("registerCommand"),
		registerShortcut: unsupported("registerShortcut"),
		registerFlag: unsupported("registerFlag"),
		getFlag: unsupported("getFlag"),
		registerMessageRenderer: unsupported("registerMessageRenderer"),
		sendMessage: unsupported("sendMessage"),
		sendUserMessage: unsupported("sendUserMessage"),
	} as unknown as ExtensionAPI;

	return {
		api,
		tools,
		handlers,
		toolNames: () => tools.map((t) => t.name),
		toolCallHandler: () => handlers.find((h) => h.event === "tool_call"),
	};
}

/** A minimal synthetic `tool_call` event for a custom (Loom) tool. */
export function makeToolCallEvent(toolName: string, input: Record<string, unknown> = {}): ToolCallEvent {
	return {
		type: "tool_call",
		toolCallId: `test-${toolName}-1`,
		toolName,
		input,
	} as ToolCallEvent;
}

/**
 * A minimal `ExtensionContext` for driving a captured handler. Only `mode`/`hasUI`
 * are load-bearing for the gate (it must branch on `ctx.mode === "tui"`); the rest
 * are throwing stubs so a handler that reaches for unstubbed context is caught.
 */
export function makeCtx(mode: ExtensionContext["mode"]): ExtensionContext {
	const hasUI = mode === "tui" || mode === "rpc";
	return {
		mode,
		hasUI,
		ui: new Proxy(
			{},
			{
				get(_t, prop) {
					return () => {
						throw new Error(`MockCtx.ui.${String(prop)}() called — not supported in a structural test.`);
					};
				},
			},
		),
	} as unknown as ExtensionContext;
}

/** Resolve a possibly-async handler return to a `ToolCallEventResult | void`. */
export async function runToolCallHandler(
	captured: CapturedHandler,
	event: ToolCallEvent,
	ctx: ExtensionContext,
): Promise<ToolCallEventResult | void> {
	return (await captured.handler(event, ctx)) as ToolCallEventResult | void;
}
