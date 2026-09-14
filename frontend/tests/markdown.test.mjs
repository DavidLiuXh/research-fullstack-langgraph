import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

test("report comparison tables render as cells and preserve citations", () => {
  const html = renderToStaticMarkup(React.createElement(ReactMarkdown, {
    remarkPlugins: [remarkGfm],
    children: "| 指标 | 同比 | 来源 |\n| --- | --- | --- |\n| 销量 | -3.2% | [来源](https://example.com) |",
  }));
  assert.match(html, /<table>/);
  assert.match(html, /<td>-3.2%<\/td>/);
  assert.match(html, /href="https:\/\/example.com"/);
  const component = readFileSync(new URL("../src/components/ChatMessagesView.tsx", import.meta.url), "utf8");
  for (const tag of component.matchAll(/<ReactMarkdown\b[^>]*>/g)) {
    assert.match(tag[0], /remarkPlugins=\{\[remarkGfm\]\}/);
  }
});
