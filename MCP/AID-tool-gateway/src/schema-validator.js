import crypto from "node:crypto";
import Ajv from "ajv";

const DEFAULT_AJV_OPTIONS = {
  allErrors: true,
  allowUnionTypes: true,
  coerceTypes: true,
  useDefaults: true,
  removeAdditional: false,
  strict: false,
};

export class SchemaValidator {
  constructor(options = {}) {
    const ajvOptions = options.ajvOptions ?? {};
    this.ajv = new Ajv({ ...DEFAULT_AJV_OPTIONS, ...ajvOptions });
    this.cache = new Map();
  }

  validate(toolId, schema, args = {}) {
    if (!schema || typeof schema !== "object") {
      return args ?? {};
    }

    const validator = this.getValidator(toolId, schema);
    const payload = cloneArgs(args);
    const valid = validator(payload);
    if (!valid) {
      throw new Error(formatAjvErrors(validator.errors));
    }
    return payload;
  }

  getValidator(toolId, schema) {
    const fingerprint = fingerprintSchema(schema);
    const cached = this.cache.get(toolId);
    if (cached && cached.fingerprint === fingerprint) {
      return cached.validate;
    }

    const compiled = this.ajv.compile(cloneSchema(schema, toolId));
    this.cache.set(toolId, { fingerprint, validate: compiled });
    return compiled;
  }
}

function cloneArgs(args) {
  if (!args || typeof args !== "object") {
    return {};
  }
  return deepClone(args);
}

function cloneSchema(schema, toolId) {
  const cloned = deepClone(schema);
  if (cloned && typeof cloned === "object" && !cloned.$id) {
    cloned.$id = `tool://${toolId}`;
  }
  return cloned;
}

function deepClone(value) {
  if (typeof globalThis.structuredClone === "function") {
    return globalThis.structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

function fingerprintSchema(schema) {
  const serialized = JSON.stringify(schema);
  return crypto.createHash("sha1").update(serialized).digest("hex");
}

function formatAjvErrors(errors) {
  if (!errors?.length) {
    return "Arguments failed schema validation.";
  }
  return errors
    .slice(0, 5)
    .map((error) => {
      const path = error.instancePath || error.schemaPath || "schema";
      return `${path}: ${error.message}`;
    })
    .join("; ");
}
