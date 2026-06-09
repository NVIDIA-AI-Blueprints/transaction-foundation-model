#!/usr/bin/env node
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

// Node version gate. Loom (and Pi 0.79.0) require Node >= 22.19.
const MIN_NODE_VERSION = "22.19.0";

function parseNodeVersion(version) {
	const [major = "0", minor = "0", patch = "0"] = version.replace(/^v/, "").split(".");
	return {
		major: Number.parseInt(major, 10) || 0,
		minor: Number.parseInt(minor, 10) || 0,
		patch: Number.parseInt(patch, 10) || 0,
	};
}

function compareNodeVersions(left, right) {
	if (left.major !== right.major) return left.major - right.major;
	if (left.minor !== right.minor) return left.minor - right.minor;
	return left.patch - right.patch;
}

const current = parseNodeVersion(process.versions.node);
if (compareNodeVersions(current, parseNodeVersion(MIN_NODE_VERSION)) < 0) {
	console.error(`loom requires Node.js >= ${MIN_NODE_VERSION} (detected ${process.versions.node}).`);
	console.error("Install or switch to a newer Node release, e.g. `nvm install 22 && nvm use 22`.");
	process.exit(1);
}

const here = import.meta.dirname;
await import(pathToFileURL(resolve(here, "..", "dist", "index.js")).href);
