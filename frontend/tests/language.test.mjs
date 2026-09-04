import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync } from "node:fs";
import ts from "typescript";

// Use the installed compiler, without adding a test runner or runtime dependency.
const source = readFileSync(new URL("../src/lib/language.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } }).outputText;
const { detectUiLanguage, translate } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`);

test("Chinese question with English names stays Chinese", () => {
  assert.equal(detectUiLanguage("请调研 DeepSeek 和 OpenAI 的技术差异"), "zh");
  assert.equal(detectUiLanguage("请研究 AI"), "zh");
});

test("foreign quotations and code do not switch English UI", () => {
  assert.equal(detectUiLanguage('Explain the policy called "中国制造2025" in detail.'), "en");
  assert.equal(detectUiLanguage("Explain this code: ```python\n标题 = '你好'```"), "en");
  assert.equal(detectUiLanguage("日本の研究について教えてください"), "en");
});

test("progress counts and review controls are localized", () => {
  assert.equal(translate("zh", "Gathered {count} sources for {query}", { count: 3, query: "OpenAI policy" }), "针对 OpenAI policy 获取了 3 个来源");
  assert.equal(translate("zh", "Approve and Continue"), "通过并继续");
  assert.equal(translate("zh", "reflection_limit_reached"), "达到维度复核上限");
  assert.equal(translate("en", "Approve and Continue"), "Approve and Continue");
  assert.equal(translate("en", "reflection_limit_reached"), "dimension-reflection limit reached");
});

test("user content is substituted once and never translated", () => {
  const query = "Reflection {count} [S1]";
  assert.equal(translate("zh", "Gathered {count} sources for {query}", { count: 2, query }), `针对 ${query} 获取了 2 个来源`);
  assert.equal(translate("zh", "Unknown source name"), "Unknown source name");
});

test("new questions can switch UI language in either direction", () => {
  assert.deepEqual(["请研究市场", "Now investigate the risks", "请补充政策背景"].map(detectUiLanguage), ["zh", "en", "zh"]);
});
