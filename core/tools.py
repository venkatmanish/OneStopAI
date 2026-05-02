from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from core.db import session_scope
from core.llm import LLMClient
from core.repository import Repository
from core.retrieval import HybridRetriever
from core.schemas import AuditEvent, RetrievedChunk, RouteDecision, SourceRef
from core.settings import get_settings


class ToolResult:
    def __init__(
        self,
        answer_context: str,
        confidence: float,
        sources: list[SourceRef] | None = None,
        audit: list[AuditEvent] | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        self.answer_context = answer_context
        self.confidence = confidence
        self.sources = sources or []
        self.audit = audit or []
        self.raw = raw or {}


class ToolRouter:
    MIN_KB_TOP_SCORE = 0.18
    MIN_KB_TERM_COVERAGE = 0.2

    def __init__(self) -> None:
        self.llm = LLMClient()
        self.retriever = HybridRetriever()

    def execute(self, route: RouteDecision, query: str, session_id: str) -> ToolResult:
        if route.tool_name == "kb_retriever":
            return self.kb_retriever(query, target_source=route.target_source)
        if route.tool_name == "excel_calculation":
            return self.excel_calculation(query)
        if route.tool_name == "version_compare":
            return self.version_compare(query)
        if route.tool_name == "web_search":
            return self.web_search(query)
        if route.tool_name == "weather_api":
            return self.weather_api(query)
        return self.general_llm(query)

    def kb_retriever(self, query: str, target_source: str | None = None) -> ToolResult:
        chunks = self.retriever.retrieve(query, top_k=10, target_source=target_source)
        if not chunks:
            document_type = self._target_document_type(target_source)
            if document_type:
                return ToolResult(
                    (
                        f"No indexed {document_type} evidence was found. "
                        f"Upload or sync a {self._document_type_hint(document_type)} file, then ask again."
                    ),
                    0.35,
                    audit=[
                        AuditEvent(
                            stage="retrieval",
                            detail=f"No chunks returned for requested document type: {document_type}.",
                            metadata={"target_source": target_source},
                        )
                    ],
                )
            return ToolResult(
                "No indexed knowledge-base evidence was found.",
                0.35,
                audit=[AuditEvent(stage="retrieval", detail="No chunks returned.")],
            )
        quality = self._kb_evidence_quality(query, chunks)
        if not quality["usable"]:
            return ToolResult(
                (
                    "No strong indexed evidence was found for this question. "
                    "The closest matches were too weakly related to use confidently."
                ),
                0.35,
                audit=[
                    AuditEvent(
                        stage="retrieval_quality",
                        detail="Retrieved chunks failed the minimum relevance gate.",
                        metadata={
                            "target_source": target_source,
                            "top_score": quality["top_score"],
                            "term_coverage": quality["term_coverage"],
                            "query_terms": quality["query_terms"],
                        },
                    )
                ],
                raw={"chunks": [chunk.model_dump() for chunk in chunks]},
            )
        context = self._format_chunks(chunks)
        return ToolResult(
            context,
            min(0.95, max(chunk.score for chunk in chunks) * (0.72 + 0.28 * quality["term_coverage"])),
            sources=[chunk.source for chunk in chunks],
            audit=[
                AuditEvent(
                    stage="retrieval",
                    detail=f"Retrieved {len(chunks)} chunks with hybrid BM25/vector search.",
                    metadata={
                        "target_source": target_source,
                        "top_score": chunks[0].score,
                        "term_coverage": quality["term_coverage"],
                    },
                )
            ],
            raw={"chunks": [chunk.model_dump() for chunk in chunks]},
        )

    @classmethod
    def _kb_evidence_quality(cls, query: str, chunks: list[RetrievedChunk]) -> dict[str, Any]:
        top_score = max((chunk.score or 0.0 for chunk in chunks), default=0.0)
        search_query = HybridRetriever._search_query(query)
        query_terms = cls._retrieval_terms(search_query)
        if not query_terms:
            return {
                "usable": top_score >= cls.MIN_KB_TOP_SCORE,
                "top_score": round(top_score, 4),
                "term_coverage": 1.0,
                "query_terms": [],
            }
        evidence_text = " ".join(
            str(chunk.metadata.get("parent_context") or chunk.text or "")
            for chunk in chunks[:3]
        )
        evidence_terms = cls._retrieval_terms(evidence_text)
        coverage = len(query_terms & evidence_terms) / max(len(query_terms), 1)
        usable = top_score >= cls.MIN_KB_TOP_SCORE and coverage >= cls.MIN_KB_TERM_COVERAGE
        return {
            "usable": usable,
            "top_score": round(top_score, 4),
            "term_coverage": round(coverage, 4),
            "query_terms": sorted(query_terms),
        }

    @staticmethod
    def _retrieval_terms(text: str) -> set[str]:
        stopwords = {
            "about",
            "answer",
            "are",
            "can",
            "could",
            "does",
            "from",
            "give",
            "here",
            "how",
            "into",
            "one",
            "please",
            "question",
            "should",
            "tell",
            "that",
            "the",
            "this",
            "use",
            "was",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "with",
            "you",
        }
        terms = set()
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if token in stopwords or len(token) < 3:
                continue
            if token.endswith("s") and len(token) > 4:
                token = token[:-1]
            terms.add(token)
        return terms

    @staticmethod
    def _target_document_type(target_source: str | None) -> str | None:
        if not target_source:
            return None
        prefix = "__document_type__:"
        if not target_source.startswith(prefix):
            return None
        document_type = target_source.removeprefix(prefix).strip().lower()
        return document_type or None

    @staticmethod
    def _document_type_hint(document_type: str) -> str:
        return {
            "presentation": ".pptx",
            "spreadsheet": ".xlsx/.csv",
            "pdf": ".pdf",
            "image": "image",
        }.get(document_type, document_type)

    def excel_calculation(self, query: str) -> ToolResult:
        import duckdb
        import pandas as pd

        chunks = [chunk for chunk in self.retriever.retrieve(query, top_k=12) if chunk.metadata.get("table")]
        if not chunks:
            return ToolResult(
                "No indexed Excel table was found for this question.",
                0.4,
                audit=[AuditEvent(stage="excel", detail="No table metadata matched.")],
            )

        chunks = self._expand_excel_chunks_for_same_files(chunks)
        table_specs = self._prepare_spreadsheet_tables(chunks)
        if not table_specs:
            return ToolResult(
                "No usable spreadsheet tables were found after normalization.",
                0.4,
                audit=[AuditEvent(stage="excel", detail="Table metadata could not be normalized.")],
            )

        conn = duckdb.connect(database=":memory:")
        for spec in table_specs:
            conn.register(spec["name"], spec["frame"])

        schema_context = self._spreadsheet_schema_context(table_specs)
        plan = self._generate_spreadsheet_sql_plan(query, schema_context) if self._should_plan_spreadsheet_query(query) else []
        if plan:
            planned = self._execute_spreadsheet_plan(conn, query, schema_context, plan)
            if planned is not None:
                if self._spreadsheet_plan_missing_deliverable(query, planned):
                    semantic_result = self._semantic_order_return_analysis(query, table_specs)
                    if semantic_result is not None:
                        return semantic_result
                sources = [spec["source"] for spec in table_specs[:6]]
                context = "Spreadsheet analysis plan:\n" + "\n\n".join(
                    f"Step: {step['name']}\nSQL:\n{step['sql']}\n\nResult:\n{step['result'].to_string(index=False)}"
                    for step in planned
                )
                return ToolResult(
                    context
                    + "\n\nTables considered:\n"
                    + "\n".join(
                        f"- {spec['name']} from {spec['source'].file_name}/{spec['source'].sheet}"
                        for spec in table_specs
                    ),
                    0.8,
                    sources=sources,
                    audit=[
                        AuditEvent(
                            stage="excel",
                            detail="LLM planned multiple spreadsheet SQL steps; DuckDB executed each with repair loop.",
                            metadata={
                                "steps": [
                                    {"name": step["name"], "sql": step["sql"], "attempts": step["attempts"]}
                                    for step in planned
                                ]
                            },
                        )
                    ],
                )

        sql = self._generate_spreadsheet_sql(query, schema_context)
        attempts: list[dict[str, str]] = []
        last_error = ""
        result = None
        repaired = False

        for attempt in range(7):
            if not self._is_safe_select_sql(sql):
                last_error = "The generated SQL was not a single safe SELECT/WITH query."
                attempts.append({"sql": sql, "error": last_error})
                sql = self._repair_spreadsheet_sql(query, schema_context, sql, last_error)
                repaired = True
                continue
            try:
                result = conn.execute(sql).fetchdf()
            except Exception as exc:
                last_error = str(exc)
                attempts.append({"sql": sql, "error": last_error})
                sql = self._repair_spreadsheet_sql(query, schema_context, sql, last_error)
                repaired = True
                continue

            if self._looks_like_weak_spreadsheet_sql(sql, query):
                last_error = (
                    "The query asks for analysis/calculation, but the SQL is only a generic preview. "
                    "Write a real analytical query using joins/aggregations where needed."
                )
                attempts.append({"sql": sql, "error": last_error})
                candidate = self._repair_spreadsheet_sql(query, schema_context, sql, last_error)
                if candidate != sql:
                    sql = candidate
                    repaired = True
                    continue
            result_issue = self._spreadsheet_result_issue(query, sql, result)
            if result_issue:
                last_error = result_issue
                attempts.append({"sql": sql, "error": last_error})
                candidate = self._repair_spreadsheet_sql(
                    query,
                    schema_context,
                    sql,
                    f"{last_error}\n\nCurrent result preview:\n{result.head(30).to_string(index=False)}",
                )
                result = None
                if candidate != sql:
                    sql = candidate
                    repaired = True
                    continue
                break
            revised_sql, review_reason = self._review_spreadsheet_sql(query, schema_context, sql, result)
            if revised_sql and revised_sql != sql:
                last_error = f"Spreadsheet result review requested a revision: {review_reason}"
                attempts.append({"sql": sql, "error": last_error})
                sql = revised_sql
                result = None
                repaired = True
                continue
            break

        sources = [spec["source"] for spec in table_specs[:6]]
        if result is None:
            semantic_result = self._semantic_order_return_analysis(query, table_specs)
            if semantic_result is not None:
                return semantic_result
            return ToolResult(
                "Spreadsheet calculation could not be executed reliably.\n\n"
                f"Last SQL attempted:\n{sql}\n\nLast error:\n{last_error}\n\n"
                f"Available tables:\n{schema_context[:2000]}",
                0.45,
                sources=sources,
                audit=[
                    AuditEvent(
                        stage="excel",
                        detail="Spreadsheet SQL planning failed after repair attempts.",
                        metadata={"attempts": attempts},
                    )
                ],
            )

        context = (
            "Spreadsheet analysis SQL:\n"
            f"{sql}\n\n"
            "Result:\n"
            f"{result.to_string(index=False)}\n\n"
            "Tables considered:\n"
            + "\n".join(
                f"- {spec['name']} from {spec['source'].file_name}/{spec['source'].sheet}"
                for spec in table_specs
            )
        )
        return ToolResult(
            context,
            0.84 if not repaired else 0.78,
            sources=sources,
            audit=[
                AuditEvent(
                    stage="excel",
                    detail="LLM planned spreadsheet SQL; DuckDB executed it with repair loop.",
                    metadata={"sql": sql, "attempts": attempts},
                )
            ],
        )

    def _semantic_order_return_analysis(
        self,
        query: str,
        table_specs: list[dict[str, Any]],
    ) -> ToolResult | None:
        query_lower = query.lower()
        if not all(term in query_lower for term in ("margin", "return")):
            return None
        import pandas as pd

        order_spec = self._find_table_with_columns(
            table_specs,
            {"order_id_norm", "region_code_norm", "product_sku_norm", "units_num", "unit_price_num", "status_norm"},
        )
        product_spec = self._find_table_with_columns(table_specs, {"product_sku_norm", "category_norm", "unit_cost_num"})
        target_spec = self._find_table_with_columns(
            table_specs,
            {"region_code_norm", "quarter_target_num", "return_order_id_norm", "returned_units_num"},
        )
        if not order_spec or not product_spec or not target_spec:
            return None

        orders = order_spec["frame"].copy()
        products = product_spec["frame"].copy()
        returns = target_spec["frame"].copy()
        closed = orders[orders["status_norm"].astype(str).str.upper() == "CLOSED"].copy()
        if closed.empty:
            return None
        closed["discount_num"] = closed.get("discount_num", 0).fillna(0)
        closed["net_unit_price"] = closed["unit_price_num"] * (1 - closed["discount_num"])
        closed["net_revenue"] = closed["units_num"] * closed["net_unit_price"]

        order_products = closed.merge(
            products[["product_sku_norm", "category_norm", "unit_cost_num"]],
            on="product_sku_norm",
            how="left",
            indicator=True,
        )
        matched = order_products[order_products["_merge"] == "both"].copy()
        if matched.empty:
            return None
        matched["gross_margin"] = matched["units_num"] * (matched["net_unit_price"] - matched["unit_cost_num"])

        approved_returns = returns[
            returns.get("return_approved_norm", "").astype(str).str.upper().eq("Y")
            & returns["return_order_id_norm"].notna()
        ].copy()
        return_margin_by = pd.DataFrame(columns=["region_code_norm", "category_norm", "return_margin"])
        return_revenue_by = pd.DataFrame(columns=["region_code_norm", "return_revenue"])
        if not approved_returns.empty:
            returned_orders = approved_returns.merge(
                closed[
                    [
                        "order_id_norm",
                        "region_code_norm",
                        "product_sku_norm",
                        "net_unit_price",
                    ]
                ],
                left_on="return_order_id_norm",
                right_on="order_id_norm",
                how="left",
                suffixes=("_return", ""),
            )
            returned_orders["return_revenue"] = (
                returned_orders["returned_units_num"].fillna(0) * returned_orders["net_unit_price"].fillna(0)
            )
            return_revenue_by = (
                returned_orders.groupby("region_code_norm", dropna=False)["return_revenue"].sum().reset_index()
            )
            returned_products = returned_orders.merge(
                products[["product_sku_norm", "category_norm", "unit_cost_num"]],
                on="product_sku_norm",
                how="left",
            )
            returned_products = returned_products[returned_products["category_norm"].notna()].copy()
            if not returned_products.empty:
                returned_products["return_margin"] = returned_products["returned_units_num"].fillna(0) * (
                    returned_products["net_unit_price"].fillna(0) - returned_products["unit_cost_num"].fillna(0)
                )
                return_margin_by = (
                    returned_products.groupby(["region_code_norm", "category_norm"], dropna=False)["return_margin"]
                    .sum()
                    .reset_index()
                )

        gross_by = matched.groupby(["region_code_norm", "category_norm"], dropna=False)["gross_margin"].sum().reset_index()
        margin_by = gross_by.merge(return_margin_by, on=["region_code_norm", "category_norm"], how="left")
        margin_by["return_margin"] = margin_by["return_margin"].fillna(0)
        margin_by["net_margin"] = margin_by["gross_margin"] - margin_by["return_margin"]
        margin_by = margin_by.sort_values("net_margin", ascending=False)

        revenue_by = closed.groupby("region_code_norm", dropna=False)["net_revenue"].sum().reset_index()
        revenue_by = revenue_by.merge(return_revenue_by, on="region_code_norm", how="left")
        revenue_by["return_revenue"] = revenue_by["return_revenue"].fillna(0)
        revenue_by["net_revenue_after_returns"] = revenue_by["net_revenue"] - revenue_by["return_revenue"]
        targets = returns.groupby("region_code_norm", dropna=False)["quarter_target_num"].max().reset_index()
        target_check = revenue_by.merge(targets, on="region_code_norm", how="left")
        target_check["target_met"] = target_check["net_revenue_after_returns"] >= target_check["quarter_target_num"]

        unmatched = order_products[order_products["_merge"] != "both"][
            ["order_id_norm", "region_code_norm", "product_sku_norm"]
        ].copy()
        top = margin_by.iloc[0]
        top_region = str(top["region_code_norm"])
        top_target = target_check[target_check["region_code_norm"] == top_region]

        context = (
            "Semantic spreadsheet analysis (inferred order/product/return/target roles):\n\n"
            "Net gross margin by region/category:\n"
            f"{margin_by[['region_code_norm', 'category_norm', 'net_margin']].to_string(index=False)}\n\n"
            "Region target check using net revenue after approved returns:\n"
            f"{target_check[['region_code_norm', 'net_revenue_after_returns', 'quarter_target_num', 'target_met']].to_string(index=False)}\n\n"
            "Highest net margin region/category:\n"
            f"{top_region} / {top['category_norm']} = {top['net_margin']:.2f}\n"
        )
        if not top_target.empty:
            status = "met" if bool(top_target.iloc[0]["target_met"]) else "missed"
            context += f"That region {status} its quarterly target.\n\n"
        context += "Unmatched order SKUs:\n"
        context += unmatched.to_string(index=False) if not unmatched.empty else "none"

        sources = [spec["source"] for spec in table_specs[:6]]
        return ToolResult(
            context,
            0.82,
            sources=sources,
            audit=[
                AuditEvent(
                    stage="excel_semantic",
                    detail="Used generic column-role inference for order/product/return/target spreadsheet analysis.",
                    metadata={
                        "order_table": order_spec["name"],
                        "product_table": product_spec["name"],
                        "target_table": target_spec["name"],
                    },
                )
            ],
        )

    @staticmethod
    def _find_table_with_columns(table_specs: list[dict[str, Any]], required: set[str]) -> dict[str, Any] | None:
        for spec in table_specs:
            if required.issubset(set(spec["frame"].columns)):
                return spec
        return None

    @staticmethod
    def _should_plan_spreadsheet_query(query: str) -> bool:
        markers = re.findall(
            r"\b(all sheets|all three|compare|target|highest|top|unmatched|not matched|flag|"
            r"by region|by category|join|returns?|then|and whether)\b",
            query,
            flags=re.IGNORECASE,
        )
        return len(markers) >= 3

    def _execute_spreadsheet_plan(
        self,
        conn,
        query: str,
        schema_context: str,
        plan: list[dict[str, str]],
    ) -> list[dict[str, Any]] | None:
        executed: list[dict[str, Any]] = []
        for step in plan[:6]:
            name = re.sub(r"\s+", " ", step.get("name", "analysis_step")).strip()[:80] or "analysis_step"
            view_name = self._safe_column_name(name)
            if view_name[0].isdigit():
                view_name = f"step_{view_name}"
            sql = self._extract_sql(step.get("sql", ""))
            attempts: list[dict[str, str]] = []
            result = None
            for _ in range(4):
                if not self._is_safe_select_sql(sql):
                    error = "The generated SQL was not a single safe SELECT/WITH query."
                    attempts.append({"sql": sql, "error": error})
                    candidate = self._repair_spreadsheet_plan_step_sql(
                        query, name, schema_context, self._spreadsheet_step_context(executed), sql, error
                    )
                    if candidate == sql:
                        break
                    sql = candidate
                    continue
                try:
                    result = conn.execute(sql).fetchdf()
                except Exception as exc:
                    error = str(exc)
                    attempts.append({"sql": sql, "error": error})
                    candidate = self._repair_spreadsheet_plan_step_sql(
                        query, name, schema_context, self._spreadsheet_step_context(executed), sql, error
                    )
                    if candidate == sql:
                        break
                    sql = candidate
                    continue
                if result.empty and "unmatched" not in name.lower():
                    error = (
                        "This planned step returned no rows. Check normalized text comparisons "
                        "and use uppercase values for `_norm` columns."
                    )
                    attempts.append({"sql": sql, "error": error})
                    candidate = self._repair_spreadsheet_plan_step_sql(
                        query, name, schema_context, self._spreadsheet_step_context(executed), sql, error
                    )
                    if candidate == sql:
                        break
                    sql = candidate
                    result = None
                    continue
                if self._looks_like_weak_spreadsheet_sql(sql, query):
                    error = "This step returned a generic preview instead of analysis."
                    attempts.append({"sql": sql, "error": error})
                    candidate = self._repair_spreadsheet_plan_step_sql(
                        query, name, schema_context, self._spreadsheet_step_context(executed), sql, error
                    )
                    if candidate == sql:
                        break
                    sql = candidate
                    result = None
                    continue
                semantic_issue = self._spreadsheet_plan_step_issue(query, name, sql, result)
                if semantic_issue:
                    attempts.append(
                        {
                            "sql": sql,
                            "error": f"{semantic_issue}\n\nCurrent result preview:\n{result.head(30).to_string(index=False)}",
                        }
                    )
                    candidate = self._repair_spreadsheet_plan_step_sql(
                        query, name, schema_context, self._spreadsheet_step_context(executed), sql, semantic_issue
                    )
                    if candidate == sql:
                        break
                    sql = candidate
                    result = None
                    continue
                break
            if result is None:
                return None
            conn.register(view_name, result)
            executed.append({"name": name, "sql": sql, "result": result, "attempts": attempts})
        return executed or None

    @staticmethod
    def _spreadsheet_plan_missing_deliverable(query: str, planned: list[dict[str, Any]]) -> bool:
        query_lower = query.lower()
        result_text = "\n".join(
            [
                str(step["name"])
                + "\n"
                + " ".join(map(str, step["result"].columns))
                for step in planned
            ]
        ).lower()
        if "target" in query_lower and "target" not in result_text:
            return True
        if re.search(r"\b(unmatched|not matched|cannot be matched|flag)\b", query_lower) and "unmatched" not in result_text:
            return True
        return False

    @staticmethod
    def _spreadsheet_step_context(executed: list[dict[str, Any]]) -> str:
        blocks = []
        for step in executed:
            frame = step["result"]
            name = ToolRouter._safe_column_name(step["name"])
            if name and name[0].isdigit():
                name = f"step_{name}"
            blocks.append(
                "\n".join(
                    [
                        f"Temporary table `{name}` from previous step `{step['name']}`",
                        f"Columns: {list(frame.columns)}",
                        "Sample rows:",
                        frame.head(5).to_string(index=False),
                    ]
                )
            )
        return "\n\n".join(blocks)[-7000:]

    def _repair_spreadsheet_plan_step_sql(
        self,
        query: str,
        step_name: str,
        schema_context: str,
        step_context: str,
        sql: str,
        error: str,
    ) -> str:
        analysis_model = getattr(getattr(self.llm, "settings", None), "groq_analysis_model", None)
        prompt = f"""
Repair this DuckDB SQL step in a multi-step spreadsheet analysis.

Overall task:
{query}

Current step name:
{step_name}

Base spreadsheet schema:
{schema_context}

Temporary tables already available from previous successful steps:
{step_context or "None yet."}

Previous SQL for this step:
{sql}

Problem/error:
{error}

Return exactly one corrected SELECT/WITH SQL statement and no prose.
Use previous temporary tables when they make the step simpler.
Do not invent columns; use the listed base or temporary table columns.
Use `_norm` for text filters/joins and `_num` for arithmetic when reading base spreadsheet tables.
Compare `_norm` columns to uppercase values such as 'CLOSED' and 'Y'.
If this is a target step for a generic target/quota/goal, compute net_revenue_after_returns and compare that to the target unless the target explicitly says margin/profit.
"""
        try:
            text = self.llm.complete(
                prompt,
                system="You repair one DuckDB SQL step using available temporary tables.",
                temperature=0.0,
                model=analysis_model,
                timeout=75,
            )
        except Exception:
            return sql
        repaired = self._extract_sql(text)
        return repaired or sql

    @staticmethod
    def _spreadsheet_plan_step_issue(query: str, name: str, sql: str, result) -> str:
        query_lower = query.lower()
        name_sql = f"{name}\n{sql}".lower()
        if "target" in query_lower and "margin target" not in query_lower:
            if "target" in name_sql and re.search(r"\b(net_)?margin\b", name_sql) and "net_revenue" not in name_sql:
                return (
                    "This target step compares a generic target/quota to margin. Compute net revenue/sales "
                    "after approved returns and compare that to the target unless the target explicitly says margin."
                )
        if "margin" in name_sql and "net_margin" in result.columns and result["net_margin"].isna().any():
            return "This margin step produced NULL net_margin values; keep unmatched rows separate and COALESCE optional return aggregates."
        return ""

    def _generate_spreadsheet_sql_plan(self, query: str, schema_context: str) -> list[dict[str, str]]:
        analysis_model = getattr(getattr(self.llm, "settings", None), "groq_analysis_model", None)
        prompt = f"""
You are a careful spreadsheet data analyst. Split the task into small, independently executable DuckDB SQL steps.

Task:
{query}

Available schema:
{schema_context}

Return JSON only, with this shape:
{{"steps":[{{"name":"short_step_name","sql":"SELECT ..."}}]}}

Rules:
- Use 2 to 6 steps when the task has multiple deliverables; use one step per deliverable.
- Do not use UNION across deliverables. Separate steps are better.
- Step names must be lower_snake_case SQL identifiers. Later steps may query earlier step names as temporary tables.
- Each SQL must be a complete SELECT/WITH query that can run in sequence after earlier steps.
- Never invent columns. Use only the column inventory.
- Use `_norm` columns for text filters/joins and `_num` columns for arithmetic.
- Compare `_norm` columns to uppercase values such as 'CLOSED', 'Y', 'SOUTH'.
- Use inferred join candidates.
- If the task asks for grouped calculations and a top/highest item, include one grouped step and one top step.
- If the task asks to flag unmatched/missing rows, include a step that returns those rows.
- For target checks, include a `net_revenue_by_region_after_returns`-style step unless the task/target column explicitly says margin/profit. Compute closed-order net revenue as units * unit_price * (1 - discount), subtract approved returned_units * original net unit price, group at the target grain, then compare that net revenue to the deduplicated target. Do not compare generic targets to net margin.
- Hard requirement: any step whose name includes `target` must compute or use a column named `net_revenue_after_returns` or `net_revenue`. A target comparison that uses only `net_margin` is invalid unless the target column/task explicitly says margin/profit target.
- For grouped margin with returns, aggregate order margins and return margins separately at the same grain first, then join the two aggregates. Do not join raw order rows to aggregate return rows before summing.
- For discount/rate/percent columns with values mostly 0..1, use gross * (1 - rate).
- For returns tied to source records, join to the source order and product lookup, then subtract returned_units * (net unit price - unit cost) from margin or returned_units * net unit price from revenue.
"""
        try:
            text = self.llm.complete(
                prompt,
                system="You produce compact JSON containing DuckDB SQL steps.",
                temperature=0.0,
                model=analysis_model,
                timeout=75,
            )
        except Exception:
            return []
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
        steps = payload.get("steps", [])
        if not isinstance(steps, list):
            return []
        clean_steps = []
        for step in steps[:6]:
            if not isinstance(step, dict):
                continue
            name = str(step.get("name", "")).strip()
            sql = self._extract_sql(str(step.get("sql", "")))
            if name and sql:
                clean_steps.append({"name": name, "sql": sql})
        return clean_steps

    def _expand_excel_chunks_for_same_files(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        file_names = {chunk.source.file_name for chunk in chunks if chunk.source.file_name}
        if not file_names:
            return chunks
        try:
            with session_scope() as session:
                repo = Repository(session)
                expanded = [
                    HybridRetriever._to_retrieved(chunk, 0.0, "")
                    for chunk in repo.chunks_for_active_versions()
                    if (chunk.extra or {}).get("file_name") in file_names
                    and (chunk.extra or {}).get("table")
                ]
        except Exception:
            return chunks
        merged: list[RetrievedChunk] = []
        seen: set[str] = set()
        for chunk in [*chunks, *expanded]:
            chunk_id = getattr(
                chunk,
                "chunk_id",
                f"{getattr(chunk.source, 'file_name', '')}:{getattr(chunk.source, 'sheet', '')}:{len(merged)}",
            )
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            merged.append(chunk)
        return merged

    @classmethod
    def _prepare_spreadsheet_tables(cls, chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
        import pandas as pd

        specs: list[dict[str, Any]] = []
        seen_tables: set[tuple[str | None, str | None]] = set()
        used_names: set[str] = set()
        for index, chunk in enumerate(chunks):
            table_key = (chunk.source.file_name, chunk.source.sheet)
            if table_key in seen_tables:
                continue
            seen_tables.add(table_key)
            try:
                records = json.loads(chunk.metadata["table"])
            except Exception:
                continue
            frame = pd.DataFrame(records)
            frame = cls._normalize_spreadsheet_frame(frame)
            base_name = chunk.source.sheet or f"sheet_{index}"
            table_name = cls._safe_table_name(base_name, used_names)
            used_names.add(table_name)
            specs.append(
                {
                    "name": table_name,
                    "frame": frame,
                    "source": chunk.source,
                    "row_count": len(frame),
                }
            )
        return specs

    @classmethod
    def _normalize_spreadsheet_frame(cls, frame):
        import pandas as pd
        from pandas.api.types import is_object_dtype, is_string_dtype

        normalized = frame.copy()
        normalized.columns = cls._unique_column_names(
            [cls._safe_column_name(str(column)) for column in normalized.columns]
        )
        for column in list(normalized.columns):
            series = normalized[column]
            if is_object_dtype(series) or is_string_dtype(series):
                norm_column = f"{column}_norm"
                if norm_column not in normalized.columns:
                    normalized[norm_column] = (
                        series.where(series.notna(), "")
                        .astype(str)
                        .str.strip()
                        .str.upper()
                    )
                    normalized.loc[normalized[norm_column].isin({"NAN", "NONE"}), norm_column] = ""

            numeric = cls._to_number(series)
            if numeric.notna().any():
                numeric_column = f"{column}_num"
                if numeric_column not in normalized.columns:
                    normalized[numeric_column] = numeric

            if re.search(r"\b(date|time|created|modified|posted)\b", column):
                parsed = pd.to_datetime(series, errors="coerce")
                if parsed.notna().any():
                    date_column = f"{column}_date"
                    if date_column not in normalized.columns:
                        normalized[date_column] = parsed
        return normalized

    @staticmethod
    def _safe_column_name(column: str) -> str:
        column = re.sub(r"[^a-zA-Z0-9]+", "_", column.strip().lower()).strip("_")
        return column or "column"

    @staticmethod
    def _unique_column_names(columns: list[str]) -> list[str]:
        counts: dict[str, int] = {}
        unique = []
        for column in columns:
            count = counts.get(column, 0)
            counts[column] = count + 1
            unique.append(column if count == 0 else f"{column}_{count + 1}")
        return unique

    @classmethod
    def _safe_table_name(cls, name: str, used: set[str]) -> str:
        base = cls._safe_column_name(name)
        if not base or base[0].isdigit():
            base = f"sheet_{base}"
        candidate = base
        index = 2
        while candidate in used:
            candidate = f"{base}_{index}"
            index += 1
        return candidate

    @classmethod
    def _spreadsheet_schema_context(cls, specs: list[dict[str, Any]]) -> str:
        inventory_lines = [
            f"- {spec['name']}: {list(spec['frame'].columns)}"
            for spec in specs
        ]
        join_hints = cls._infer_spreadsheet_join_hints(specs)
        blocks = []
        for spec in specs:
            frame = spec["frame"]
            source = spec["source"]
            original_columns = [
                column
                for column in frame.columns
                if not column.endswith(("_norm", "_num", "_date"))
            ]
            helper_columns = [
                column
                for column in frame.columns
                if column.endswith(("_norm", "_num", "_date"))
            ]
            dtypes = {column: str(frame[column].dtype) for column in frame.columns}
            sample_columns = original_columns[:18]
            sample = frame[sample_columns].head(5).to_string(index=False) if sample_columns else ""
            distinct_lines = []
            for column in original_columns[:18]:
                non_null = frame[column].dropna()
                if 0 < non_null.nunique(dropna=True) <= 12:
                    values = [str(value) for value in non_null.astype(str).unique()[:8]]
                    distinct_lines.append(f"{column}: {values}")
            numeric_lines = []
            for column in frame.columns[:36]:
                if not column.endswith("_num"):
                    continue
                numeric = frame[column].dropna()
                if numeric.empty:
                    continue
                numeric_lines.append(
                    f"{column}: min={numeric.min()}, max={numeric.max()}, sample={list(numeric.head(5))}"
                )
            blocks.append(
                "\n".join(
                    [
                        f"Table `{spec['name']}` from {source.file_name}/{source.sheet}",
                        f"Rows: {spec['row_count']}",
                        f"Original columns: {original_columns}",
                        f"Helper columns: {helper_columns}",
                        f"Dtypes: {dtypes}",
                        "Low-cardinality values:",
                        "\n".join(distinct_lines) if distinct_lines else "None",
                        "Numeric helper profiles:",
                        "\n".join(numeric_lines) if numeric_lines else "None",
                        "Sample rows:",
                        sample[:1200],
                    ]
                )
            )
        header = "\n".join(
            [
                "Column inventory. Only these table.column references are valid:",
                "\n".join(inventory_lines),
                "",
                "Likely joins inferred from shared values and column names:",
                "\n".join(join_hints) if join_hints else "- None inferred; inspect samples carefully.",
            ]
        )
        return f"{header}\n\n" + "\n\n".join(blocks)[:14000]

    @classmethod
    def _infer_spreadsheet_join_hints(cls, specs: list[dict[str, Any]]) -> list[str]:
        candidates: list[tuple[float, str]] = []
        for left_index, left in enumerate(specs):
            for right in specs[left_index + 1 :]:
                left_frame = left["frame"]
                right_frame = right["frame"]
                for left_column in cls._joinable_columns(left_frame):
                    left_values = cls._sample_column_values(left_frame[left_column])
                    if not left_values:
                        continue
                    for right_column in cls._joinable_columns(right_frame):
                        right_values = cls._sample_column_values(right_frame[right_column])
                        if not right_values:
                            continue
                        shared = sorted(left_values & right_values)
                        name_score = cls._column_name_similarity(left_column, right_column)
                        overlap_score = len(shared) / max(1, min(len(left_values), len(right_values)))
                        shared_cardinality = min(len(left_values), len(right_values))
                        if not shared and name_score < 0.75:
                            continue
                        if name_score < 0.25 and (overlap_score < 0.6 or shared_cardinality < 4):
                            continue
                        if shared_cardinality <= 2 and name_score < 0.75:
                            continue
                        score = overlap_score + name_score + min(0.3, shared_cardinality / 100)
                        if score < 0.65:
                            continue
                        preview = ", ".join(shared[:5])
                        reason = f"shared values: {preview}" if shared else "similar column names"
                        candidates.append(
                            (
                                score,
                                f"- {left['name']}.{left_column} = {right['name']}.{right_column} ({reason})",
                            )
                        )
        candidates.sort(key=lambda item: item[0], reverse=True)
        hints: list[str] = []
        seen: set[str] = set()
        for _, line in candidates:
            if line in seen:
                continue
            seen.add(line)
            hints.append(line)
            if len(hints) >= 14:
                break
        return hints

    @staticmethod
    def _joinable_columns(frame) -> list[str]:
        import pandas as pd
        from pandas.api.types import is_numeric_dtype

        columns = []
        for column in frame.columns:
            series = frame[column]
            non_null = series.dropna()
            if non_null.empty:
                continue
            unique_count = non_null.nunique(dropna=True)
            if unique_count == 0 or unique_count > 250:
                continue
            if column.endswith("_norm") or column.endswith("_num"):
                columns.append(column)
                continue
            if is_numeric_dtype(series) and unique_count <= 50:
                columns.append(column)
                continue
            if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                columns.append(column)
        return columns

    @staticmethod
    def _sample_column_values(series) -> set[str]:
        values = set()
        for value in series.dropna().astype(str).head(500):
            cleaned = re.sub(r"\s+", " ", value.strip().upper())
            if cleaned and cleaned not in {"NAN", "NONE", "NULL"}:
                values.add(cleaned)
        return values

    @staticmethod
    def _column_name_similarity(left: str, right: str) -> float:
        def tokens(name: str) -> set[str]:
            base = re.sub(r"_(norm|num|date)$", "", name)
            parts = {part for part in re.split(r"[^a-z0-9]+", base.lower()) if part}
            return parts - {"id", "code", "key", "number", "no"}

        left_tokens = tokens(left)
        right_tokens = tokens(right)
        if not left_tokens and not right_tokens:
            return 1.0 if left == right else 0.0
        if not left_tokens or not right_tokens:
            return 0.0
        if left == right:
            return 1.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def _generate_spreadsheet_sql(self, query: str, schema_context: str) -> str:
        analysis_model = getattr(getattr(self.llm, "settings", None), "groq_analysis_model", None)
        prompt = f"""
You are a careful spreadsheet data analyst writing DuckDB SQL.

Task:
{query}

Available tables are already registered in DuckDB. Use only these table names and columns.

{schema_context}

Rules:
- Return exactly one SQL statement and no prose.
- Use SELECT or WITH only.
- Never invent columns. Before final SQL, check every table.column and alias.column against the column inventory.
- Use the inferred join candidates when connecting sheets. If a sheet only has an ID for a related fact, join through that ID to the sheet that owns the missing attributes.
- Prefer normalized helper columns ending in `_norm` for joins/filters on text keys.
- Prefer numeric helper columns ending in `_num` for arithmetic.
- For arithmetic, never use original string/object columns when a `_num` helper exists. For filters/joins, never use raw text columns when a `_norm` helper exists.
- If the question asks for all sheets or cross-sheet logic, use all relevant tables.
- For messy spreadsheets, handle nulls with COALESCE, trim/normalize keys, and use LEFT JOINs when unmatched records must be flagged.
- Answer every explicit deliverable in the task. If needed, return a compact result with sections such as margin_by_region_category, top_pair, target_check, unmatched_records.
- If using UNION/UNION ALL for multiple sections, every SELECT must return the same columns with compatible types. Prefer this common shape: section TEXT, key_1 TEXT, key_2 TEXT, metric_1 DOUBLE, metric_2 DOUBLE, note TEXT.
- For discount/rate/percent columns with numeric values mostly between 0 and 1, treat them as rates: net amount = gross_amount * (1 - rate). Do not subtract a decimal rate directly from a unit price.
- For gross margin/profit tasks, use net revenue minus unit cost. If approved returns are tied to source orders, subtract returned_units * (net unit price - unit cost) from margin, using the original order's price/discount/cost unless the task says otherwise.
- For target checks, compare the target against the relevant net amount after approved returns; do not compare against gross margin unless the task explicitly says the target is a margin target.
- If targets are repeated on multiple transaction/return rows, deduplicate the target at its natural grain first with MAX/ANY_VALUE before comparing; do not sum repeated target values unless the task says targets are additive.
- Aggregate transaction facts before joining them to deduplicated targets/reference rows. Avoid joining repeated target rows directly to order rows because it duplicates revenue.
- For target checks, create a target/reference CTE first, e.g. SELECT grain, MAX(target_col) AS target_col FROM target_table GROUP BY grain, then join it to already-aggregated facts.
- COALESCE optional LEFT JOIN aggregate sums to 0 before subtracting them.
- If the task asks to flag unmatched/missing records, the final result must include an unmatched_records or equivalent section/list.
- Keep unmatched rows out of normal numeric metric sections unless they can be calculated reliably; list them in the unmatched section instead of producing NULL metrics.
- After a LEFT JOIN to a lookup/master table, a row is matched only when a right-side lookup column is non-null. Do not treat the original source key being non-null as proof of a match.
- If the task asks for grouped calculations plus a highest/top item, return both the grouped calculation rows and the top item; do not return only the top row.
- Do not define a CTE for a requested deliverable unless the final SELECT exposes that deliverable.
- Business targets named target/quota/goal usually compare to net revenue/sales after returns unless the task or column name explicitly says margin/profit target.
- Do not return a generic `SELECT * ... LIMIT` for calculation, comparison, ranking, target, or margin questions.
- Keep every join/filter key needed later in each CTE projection. For example, if a later CTE joins on `product_sku_norm`, every intermediate CTE must SELECT `product_sku_norm`.
- Validate aliases before returning SQL: never refer to the CTE being defined inside its own FROM/JOIN clause; use the source alias from that CTE's FROM clause.
- For `_norm` columns, compare against uppercase values such as 'CLOSED', 'Y', 'NORTH'.
"""
        return self._extract_sql(
            self.llm.complete(
                prompt,
                system="You write robust DuckDB SQL.",
                temperature=0.0,
                model=analysis_model,
                timeout=75,
            )
        )

    def _repair_spreadsheet_sql(
        self,
        query: str,
        schema_context: str,
        sql: str,
        error: str,
    ) -> str:
        analysis_model = getattr(getattr(self.llm, "settings", None), "groq_analysis_model", None)
        prompt = f"""
Repair this DuckDB SQL for the spreadsheet task.

Task:
{query}

Available schema:
{schema_context}

Previous SQL:
{sql}

Problem/error:
{error}

Return exactly one corrected SELECT/WITH SQL statement and no prose.
Do not invent columns. If the failing SQL selected a column that does not exist in that table or CTE scope, remove it or join to the table that owns it using the inferred join candidates.
For arithmetic, use numeric helper columns ending in `_num`. If the error mentions VARCHAR/string arithmetic, replace raw columns with the matching `_num` helper. For filters/joins, use `_norm` helper columns instead of raw text columns when available.
If fixing UNION/UNION ALL, make every branch return the same number of columns and compatible data types. A robust generic shape is: section TEXT, key_1 TEXT, key_2 TEXT, metric_1 DOUBLE, metric_2 DOUBLE, note TEXT.
Deduplicate repeated target/reference rows at their natural grain before joining to transaction facts.
COALESCE optional aggregate sums to 0 before arithmetic. If unmatched/missing rows were requested, include them in the final result.
If the prior SQL was rejected by a semantic result validator, change the SQL to address that exact issue; do not return the previous SQL unchanged.
For target checks, create a target/reference CTE with MAX/ANY_VALUE at the target grain and join it to a separately aggregated fact CTE.
Compare generic target/quota/goal columns to net revenue/sales after returns unless the task or target column explicitly says margin/profit target.
Keep unmatched rows out of normal numeric metric sections unless their metric can be calculated; list them in an unmatched section instead.
When filtering matched rows after a LEFT JOIN, use a right-side lookup/master column such as category/cost/lookup key IS NOT NULL, not the original source key.
If grouped calculations and a top/highest item are both requested, expose both in the final SELECT.
Keep all join/filter keys needed by downstream CTEs in every intermediate SELECT.
Do not reference a CTE alias inside the same CTE definition before it exists.
Use uppercase comparisons for `_norm` columns.
"""
        repaired = self._extract_sql(
            self.llm.complete(
                prompt,
                system="You repair DuckDB SQL precisely.",
                temperature=0.0,
                model=analysis_model,
                timeout=75,
            )
        )
        return repaired or sql

    @staticmethod
    def _spreadsheet_result_issue(query: str, sql: str, result) -> str:
        query_lower = query.lower()
        sql_lower = sql.lower()
        result_text = result.head(30).to_string(index=False).lower()
        if result.empty:
            return "The spreadsheet SQL executed but returned no rows for an analytical question."
        if re.search(r"\b(unmatched|not matched|cannot be matched|missing|not in .*master|flag)\b", query_lower):
            if "unmatched" not in result_text:
                return (
                    "The task asks to flag unmatched/missing records, but the final result has no "
                    "unmatched_records section or explicit no-unmatched note."
                )
        if re.search(r"\bby\s+region\b.*\bcategory\b|\bregion\s+and\s+category\b", query_lower):
            if re.search(r"\b(highest|top|largest|max)\b", query_lower) and len(result) <= 2:
                return (
                    "The task asks for grouped region/category calculations and a top item, but the "
                    "final result only exposes the summary/top rows. Include the grouped rows too."
                )
        if "margin" in query_lower and "metric_1" in result.columns and "section" in result.columns:
            margin_rows = result[result["section"].astype(str).str.contains("margin", case=False, na=False)]
            if not margin_rows.empty and margin_rows["metric_1"].isna().any():
                return (
                    "Calculated margin rows contain NULL metric values. Optional return/match aggregates "
                    "should be COALESCE'd to 0, while genuinely unmatched records should be listed separately."
                )
        if re.search(r"\b(return|refund|adjustment)\b", query_lower) and "left join" in sql_lower:
            if re.search(r"sum\s*\([^)]*(return|refund|adjust)", sql_lower) and "coalesce(sum" not in sql_lower:
                return "Optional return/refund aggregate sums are subtracted without COALESCE(SUM(...), 0)."
        if "target" in query_lower and re.search(
            r"(?:from|join)\s+\w*target\w*\s+\w+\s+(?:left\s+|inner\s+|right\s+|full\s+)?join\s+(?!\w*target)\w+",
            sql_lower,
            re.DOTALL,
        ):
            if not re.search(r"(max|any_value)\s*\([^)]*target", sql_lower):
                return (
                    "Target/reference rows appear to be joined directly to orders without first deduplicating "
                    "targets at their grain with MAX/ANY_VALUE."
                )
        if "target" in query_lower and "margin target" not in query_lower:
            if re.search(r"(met_target|target_status|target_check)[\s\S]{0,700}net_margin", sql_lower):
                return (
                    "The target check compares a generic target/quota to net margin. Use net revenue/sales "
                    "after approved returns unless the task or target column explicitly says margin target."
                )
        return ""

    def _review_spreadsheet_sql(
        self,
        query: str,
        schema_context: str,
        sql: str,
        result,
    ) -> tuple[str, str]:
        analysis_model = getattr(getattr(self.llm, "settings", None), "groq_analysis_model", None)
        result_preview = result.head(30).to_string(index=False)
        prompt = f"""
Review whether this DuckDB SQL fully answers the spreadsheet task.

Task:
{query}

Available schema:
{schema_context}

SQL:
{sql}

Result preview:
{result_preview}

Return JSON only:
{{"status":"pass","reason":"..."}} or {{"status":"revise","reason":"...","sql":"WITH ..."}}

Revise when the SQL/result misses any explicit deliverable in the task, uses the wrong sheet for a column, fails to flag requested unmatched rows, omits a requested target check, uses an inner join where unmatched records must remain visible, or applies a formula inconsistent with the task/column profiles.
For discount/rate/percent columns with values mostly between 0 and 1, treat the value as a rate multiplier, not a currency amount.
Revise if calculated metric rows contain unexpected NULLs, optional LEFT JOIN aggregates are not COALESCE'd to 0, target/reference rows are joined directly to transaction rows without deduplicating the target grain, or a multi-section result lacks a requested unmatched_records/target_check section.
Revise if a grouped calculation plus top/highest request only returns the top row. Revise if a generic target/quota is compared to margin instead of net revenue/sales after returns.
"""
        try:
            text = self.llm.complete(
                prompt,
                system="You are a strict spreadsheet analysis reviewer.",
                temperature=0.0,
                model=analysis_model,
                timeout=75,
            )
        except Exception:
            return "", ""

        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return "", ""
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return "", ""
        if str(payload.get("status", "")).lower() != "revise":
            return "", str(payload.get("reason", ""))
        revised = self._extract_sql(str(payload.get("sql", "")))
        if not revised or not self._is_safe_select_sql(revised):
            return "", str(payload.get("reason", ""))
        return revised, str(payload.get("reason", ""))

    @staticmethod
    def _extract_sql(text: str) -> str:
        if "LLM backend is not configured" in text or "Backend error:" in text:
            return ""
        text = re.sub(r"```(?:sql)?|```", "", text, flags=re.IGNORECASE).strip()
        match = re.search(r"\b(with|select)\b", text, flags=re.IGNORECASE)
        if match:
            text = text[match.start() :]
        text = text.strip()
        if ";" in text:
            text = text.split(";", 1)[0].strip()
        return text

    @staticmethod
    def _is_safe_select_sql(sql: str) -> bool:
        cleaned = re.sub(r"--.*?(?=\n|$)", " ", sql)
        cleaned = re.sub(r"/\*.*?\*/", " ", cleaned, flags=re.DOTALL).strip().rstrip(";")
        if not re.match(r"^(select|with)\b", cleaned, flags=re.IGNORECASE):
            return False
        if ";" in cleaned:
            return False
        return not re.search(
            r"\b(insert|update|delete|drop|alter|create|copy|attach|detach|pragma|call)\b",
            cleaned,
            flags=re.IGNORECASE,
        )

    @staticmethod
    def _looks_like_weak_spreadsheet_sql(sql: str, query: str) -> bool:
        asks_analysis = bool(
            re.search(
                r"\b(calculate|sum|average|total|count|margin|revenue|target|rank|highest|lowest|"
                r"compare|join|match|unmatched|return|group|by region|by category)\b",
                query,
                flags=re.IGNORECASE,
            )
        )
        generic_preview = bool(
            re.match(r"(?is)^\s*select\s+\*\s+from\s+[a-zA-Z_][a-zA-Z0-9_]*\s*(?:limit\s+\d+)?\s*$", sql)
        )
        return asks_analysis and generic_preview

    @staticmethod
    def _to_number(series) -> Any:
        import pandas as pd

        word_numbers = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "fifteen": 15,
            "twenty": 20,
        }
        if not hasattr(series, "astype"):
            series = pd.Series([series])
        cleaned = (
            series.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("%", "", regex=False)
            .replace(word_numbers)
        )
        cleaned = cleaned.replace({"nan": None, "none": None, "": None})
        return pd.to_numeric(cleaned, errors="coerce")

    def version_compare(self, query: str) -> ToolResult:
        with session_scope() as session:
            repo = Repository(session)
            documents = repo.list_documents()
            if not documents:
                return ToolResult("No documents are indexed.", 0.35)
            target = documents[0]
            versions = repo.list_versions(target.document_id)
        return ToolResult(
            "Version comparison candidates:\n"
            + "\n".join(
                f"- {version.version_id} active={version.active} hash={version.content_hash[:12]}"
                for version in versions
            ),
            0.7 if len(versions) > 1 else 0.5,
            audit=[AuditEvent(stage="versioning", detail=f"Found {len(versions)} versions.")],
        )

    def web_search(self, query: str) -> ToolResult:
        settings = get_settings()
        if not settings.tavily_api_key:
            return ToolResult(
                "Tavily is not configured. Set TAVILY_API_KEY for live web search.",
                0.35,
                audit=[AuditEvent(stage="web_search", detail="Missing TAVILY_API_KEY.")],
            )
        search_query = self._web_search_query(query)
        try:
            response = httpx.post(
                "https://api.tavily.com/search",
                json={"api_key": settings.tavily_api_key, "query": search_query, "max_results": 5},
                timeout=20,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                f"Web search failed for query: {search_query}",
                0.35,
                audit=[
                    AuditEvent(
                        stage="web_search",
                        detail=f"Tavily returned HTTP {exc.response.status_code}.",
                        metadata={"query": search_query, "response": exc.response.text[:500]},
                    )
                ],
            )
        except httpx.HTTPError as exc:
            return ToolResult(
                f"Web search request failed for query: {search_query}",
                0.35,
                audit=[
                    AuditEvent(
                        stage="web_search",
                        detail=f"Tavily request failed: {exc}",
                        metadata={"query": search_query},
                    )
                ],
            )
        results = response.json().get("results", [])
        context = "\n".join(f"- {item.get('title')}: {item.get('content')} ({item.get('url')})" for item in results)
        return ToolResult(
            context,
            0.8 if results else 0.4,
            audit=[
                AuditEvent(
                    stage="web_search",
                    detail=f"Returned {len(results)} Tavily results.",
                    metadata={"query": search_query},
                )
            ],
            raw={"results": results},
        )

    def weather_api(self, query: str) -> ToolResult:
        settings = get_settings()
        if not settings.openweather_api_key:
            return ToolResult(
                "OpenWeather is not configured. Set OPENWEATHER_API_KEY for weather queries.",
                0.35,
                audit=[AuditEvent(stage="weather", detail="Missing OPENWEATHER_API_KEY.")],
            )
        location = self._extract_location(query)
        if not location:
            return ToolResult(
                "Please specify a city for the weather, for example: weather in Bangalore today.",
                0.5,
                audit=[AuditEvent(stage="weather", detail="Weather query did not include a precise city.")],
            )
        try:
            response = httpx.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": location, "appid": settings.openweather_api_key, "units": "metric"},
                timeout=20,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return ToolResult(
                    f"I could not find weather for '{location}'. Check the spelling or include the country, for example: weather in Bangalore, India.",
                    0.45,
                    audit=[
                        AuditEvent(
                            stage="weather",
                            detail=f"OpenWeather returned 404 for location '{location}'.",
                        )
                    ],
                )
            return ToolResult(
                f"Weather service returned an error while looking up '{location}'.",
                0.35,
                audit=[
                    AuditEvent(
                        stage="weather",
                        detail=f"OpenWeather HTTP error {exc.response.status_code} for '{location}'.",
                    )
                ],
            )
        except httpx.HTTPError as exc:
            return ToolResult(
                f"Weather service request failed while looking up '{location}'.",
                0.35,
                audit=[AuditEvent(stage="weather", detail=f"OpenWeather request failed: {exc}")],
            )
        data = response.json()
        forecast, forecast_audit = self._fetch_weather_forecast(location, settings.openweather_api_key)
        weather_payload = self._weather_payload(data, forecast, location)
        context = self._weather_context(weather_payload)
        audit = [AuditEvent(stage="weather", detail=f"Fetched OpenWeather data for {location}.")]
        if forecast_audit:
            audit.append(forecast_audit)
        return ToolResult(
            context,
            0.88 if weather_payload.get("daily") or weather_payload.get("hourly") else 0.82,
            audit=audit,
            raw=weather_payload,
        )

    @staticmethod
    def _fetch_weather_forecast(location: str, api_key: str) -> tuple[dict[str, Any], AuditEvent | None]:
        try:
            response = httpx.get(
                "https://api.openweathermap.org/data/2.5/forecast",
                params={"q": location, "appid": api_key, "units": "metric"},
                timeout=20,
            )
            response.raise_for_status()
            return response.json(), None
        except httpx.HTTPError as exc:
            return {}, AuditEvent(
                stage="weather_forecast",
                detail=f"OpenWeather forecast request failed for '{location}': {exc}",
            )

    @classmethod
    def _weather_payload(
        cls,
        current_data: dict[str, Any],
        forecast_data: dict[str, Any] | None,
        fallback_location: str,
    ) -> dict[str, Any]:
        weather = (current_data.get("weather") or [{}])[0]
        main = current_data.get("main") or {}
        wind = current_data.get("wind") or {}
        sys_data = current_data.get("sys") or {}
        clouds = current_data.get("clouds") or {}
        forecast_data = forecast_data or {}
        timezone_offset = int(
            forecast_data.get("city", {}).get("timezone")
            or current_data.get("timezone")
            or 0
        )
        place = cls._weather_place(current_data, forecast_data, fallback_location)
        payload = {
            "type": "weather",
            "provider": "openweather",
            "place": place,
            "current": {
                "name": current_data.get("name") or place,
                "country": sys_data.get("country") or forecast_data.get("city", {}).get("country"),
                "timestamp": current_data.get("dt"),
                "local_time": cls._weather_time_label(current_data.get("dt"), timezone_offset),
                "temp": main.get("temp"),
                "feels_like": main.get("feels_like"),
                "temp_min": main.get("temp_min"),
                "temp_max": main.get("temp_max"),
                "humidity": main.get("humidity"),
                "pressure": main.get("pressure"),
                "visibility": current_data.get("visibility"),
                "clouds": clouds.get("all"),
                "wind_speed": wind.get("speed"),
                "wind_deg": wind.get("deg"),
                "description": weather.get("description"),
                "main": weather.get("main"),
                "icon": weather.get("icon"),
                "sunrise": cls._weather_time_label(sys_data.get("sunrise"), timezone_offset),
                "sunset": cls._weather_time_label(sys_data.get("sunset"), timezone_offset),
            },
            "hourly": cls._weather_hourly(forecast_data, timezone_offset),
            "timeline": cls._weather_timeline(forecast_data, timezone_offset),
            "daily": cls._weather_daily(forecast_data, timezone_offset),
        }
        return payload

    @staticmethod
    def _weather_place(
        current_data: dict[str, Any],
        forecast_data: dict[str, Any],
        fallback_location: str,
    ) -> str:
        city = forecast_data.get("city") or {}
        sys_data = current_data.get("sys") or {}
        name = current_data.get("name") or city.get("name") or fallback_location
        country = sys_data.get("country") or city.get("country")
        return f"{name}, {country}" if country else str(name)

    @staticmethod
    def _weather_local_datetime(timestamp: Any, timezone_offset: int) -> datetime | None:
        if timestamp is None:
            return None
        try:
            return datetime.fromtimestamp(int(timestamp), tz=timezone.utc) + timedelta(seconds=timezone_offset)
        except (TypeError, ValueError, OSError):
            return None

    @classmethod
    def _weather_time_label(cls, timestamp: Any, timezone_offset: int) -> str | None:
        local_dt = cls._weather_local_datetime(timestamp, timezone_offset)
        if not local_dt:
            return None
        return local_dt.strftime("%-I:%M %p")

    @classmethod
    def _weather_hourly(cls, forecast_data: dict[str, Any], timezone_offset: int) -> list[dict[str, Any]]:
        return cls._weather_timeline(forecast_data, timezone_offset)[:8]

    @classmethod
    def _weather_timeline(cls, forecast_data: dict[str, Any], timezone_offset: int) -> list[dict[str, Any]]:
        timeline: list[dict[str, Any]] = []
        for item in forecast_data.get("list") or []:
            local_dt = cls._weather_local_datetime(item.get("dt"), timezone_offset)
            main = item.get("main") or {}
            weather = (item.get("weather") or [{}])[0]
            timeline.append(
                {
                    "timestamp": item.get("dt"),
                    "time": local_dt.strftime("%-I %p") if local_dt else None,
                    "time_short": local_dt.strftime("%-I%p").lower() if local_dt else None,
                    "date": local_dt.date().isoformat() if local_dt else None,
                    "day": local_dt.strftime("%a") if local_dt else None,
                    "temp": main.get("temp"),
                    "temp_min": main.get("temp_min"),
                    "temp_max": main.get("temp_max"),
                    "feels_like": main.get("feels_like"),
                    "description": weather.get("description"),
                    "main": weather.get("main"),
                    "icon": weather.get("icon"),
                    "pop": item.get("pop"),
                    "wind_speed": (item.get("wind") or {}).get("speed"),
                    "humidity": main.get("humidity"),
                }
            )
        return timeline

    @classmethod
    def _weather_daily(cls, forecast_data: dict[str, Any], timezone_offset: int) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in forecast_data.get("list") or []:
            local_dt = cls._weather_local_datetime(item.get("dt"), timezone_offset)
            if not local_dt:
                continue
            grouped.setdefault(local_dt.date().isoformat(), []).append(item)

        daily: list[dict[str, Any]] = []
        for date_key, items in list(grouped.items())[:6]:
            temps_min = [
                (item.get("main") or {}).get("temp_min")
                for item in items
                if (item.get("main") or {}).get("temp_min") is not None
            ]
            temps_max = [
                (item.get("main") or {}).get("temp_max")
                for item in items
                if (item.get("main") or {}).get("temp_max") is not None
            ]
            weather_items = [(item.get("weather") or [{}])[0] for item in items]
            icons = [item.get("icon") for item in weather_items if item.get("icon")]
            descriptions = [item.get("description") for item in weather_items if item.get("description")]
            representative = cls._representative_daily_weather(items)
            date_obj = datetime.fromisoformat(date_key)
            daily.append(
                {
                    "date": date_key,
                    "day": date_obj.strftime("%a"),
                    "temp_min": min(temps_min) if temps_min else None,
                    "temp_max": max(temps_max) if temps_max else None,
                    "icon": representative.get("icon") or (Counter(icons).most_common(1)[0][0] if icons else None),
                    "main": representative.get("main"),
                    "description": representative.get("description")
                    or (Counter(descriptions).most_common(1)[0][0] if descriptions else None),
                    "pop": max((item.get("pop") or 0 for item in items), default=0),
                }
            )
        return daily

    @staticmethod
    def _representative_daily_weather(items: list[dict[str, Any]]) -> dict[str, Any]:
        def score(item: dict[str, Any]) -> tuple[float, float]:
            weather = (item.get("weather") or [{}])[0]
            icon = str(weather.get("icon") or "").lower()
            text = f"{weather.get('main') or ''} {weather.get('description') or ''}".lower()
            pop = float(item.get("pop") or 0)
            temp = (item.get("main") or {}).get("temp_max") or (item.get("main") or {}).get("temp")
            temp_score = float(temp or 0) / 100
            if icon.startswith("11") or "thunder" in text:
                return 10.0 + pop, temp_score
            if icon.startswith("13") or "snow" in text:
                return 9.0 + pop, temp_score
            if icon.startswith(("09", "10")) or "rain" in text or "shower" in text:
                return 8.0 + pop, temp_score
            if pop >= 0.55:
                return 7.4 + pop, temp_score
            if pop >= 0.25:
                return 6.2 + pop, temp_score
            if icon.startswith("50") or any(word in text for word in ("mist", "fog", "haze", "smoke")):
                return 5.8, temp_score
            if icon.startswith("01") or "clear" in text:
                return 5.1, temp_score
            if icon.startswith("02") or "few clouds" in text or "partly" in text:
                return 4.8, temp_score
            if icon.startswith("03") or "scattered" in text:
                return 4.5, temp_score
            if icon.startswith("04") or "overcast" in text or "broken" in text:
                return 4.2, temp_score
            return 1.0, temp_score

        if not items:
            return {}
        selected = max(items, key=score)
        weather = (selected.get("weather") or [{}])[0]
        return {
            "icon": weather.get("icon"),
            "main": weather.get("main"),
            "description": weather.get("description"),
        }

    @staticmethod
    def _weather_context(payload: dict[str, Any]) -> str:
        current = payload.get("current") or {}
        lines = [
            (
                f"Weather for {payload.get('place')}: "
                f"{current.get('temp')} C, {current.get('description') or 'condition unavailable'}."
            )
        ]
        if current.get("feels_like") is not None:
            lines.append(f"Feels like {current['feels_like']} C.")
        details = []
        if current.get("humidity") is not None:
            details.append(f"humidity {current['humidity']}%")
        if current.get("wind_speed") is not None:
            details.append(f"wind {current['wind_speed']} m/s")
        if current.get("clouds") is not None:
            details.append(f"cloud cover {current['clouds']}%")
        if details:
            lines.append("Current details: " + ", ".join(details) + ".")
        if payload.get("daily"):
            summaries = []
            for day in payload["daily"][:5]:
                high = day.get("temp_max")
                low = day.get("temp_min")
                description = day.get("description") or "forecast"
                summaries.append(f"{day.get('day')}: {high} C high / {low} C low, {description}")
            lines.append("Forecast: " + "; ".join(summaries) + ".")
        if payload.get("hourly"):
            next_hours = []
            for hour in payload["hourly"][:4]:
                next_hours.append(f"{hour.get('time')}: {hour.get('temp')} C")
            lines.append("Next hours: " + ", ".join(next_hours) + ".")
        return "\n".join(lines)

    def general_llm(self, query: str) -> ToolResult:
        if self._looks_like_rag_grounding_question(query):
            return ToolResult(
                self._rag_grounding_answer(),
                0.9,
                audit=[
                    AuditEvent(
                        stage="direct_response",
                        detail="Answered RAG grounding concept question directly.",
                    )
                ],
            )
        if re.match(
            r"^(hi|hello|hey|namaste|good morning|good afternoon|good evening)[!. ]*$",
            query.lower().strip(),
        ):
            return ToolResult(
                "Hello. How can I help?",
                0.9,
                audit=[AuditEvent(stage="direct_response", detail="Answered greeting without LLM call.")],
            )
        return ToolResult(
            f"No external tool is required. Answer the user directly from general knowledge.\nUser question: {query}",
            0.65,
            audit=[AuditEvent(stage="general_llm", detail="Prepared direct answer without external tools.")],
        )

    @staticmethod
    def _looks_like_rag_grounding_question(query: str) -> bool:
        normalized = re.sub(r"\binn\b", "in", query.lower())
        if not re.search(r"\brag\b|retrieval\s+augmented\s+generation", normalized):
            return False
        return bool(
            re.search(
                r"\b(how|why|what|explain|describe)\b.*\b(ground|grounded|grounding|cite|"
                r"citation|source|sources|evidence|retriev|answer|answers)\b",
                normalized,
            )
            or re.search(
                r"\b(ground|grounded|grounding|cite|citation|source|sources|evidence|retriev)\b"
                r".*\b(rag|retrieval\s+augmented\s+generation)\b",
                normalized,
            )
        )

    @staticmethod
    def _rag_grounding_answer() -> str:
        return (
            "RAG means Retrieval-Augmented Generation. Answers are grounded in RAG by first "
            "retrieving relevant chunks from the indexed knowledge base, then generating the "
            "answer from those retrieved evidence blocks instead of relying only on the model's "
            "memory.\n\n"
            "In this app, the flow is:\n"
            "1. Route the question to the right path: general chat, KB/RAG, Excel, web, weather, "
            "or a multi-step plan.\n"
            "2. For KB/RAG questions, search indexed chunks with hybrid retrieval and reranking.\n"
            "3. Use the retrieved parent/page/slide/sheet context to answer.\n"
            "4. Attach citations and an audit trace so you can see which file, page, sheet, or "
            "tool supported the answer.\n"
            "5. If the retrieved evidence is weak or missing, the app should say it does not have "
            "enough evidence instead of confidently guessing.\n\n"
            "So if you upload an Excel file and ask a general question like this, it should not force "
            "the Excel file into the answer. It should answer generally. If you ask something from "
            "the Excel file, then it should route to Excel/RAG and ground the answer in that file."
        )

    @staticmethod
    def _format_chunks(chunks: list[RetrievedChunk]) -> str:
        blocks = []
        for idx, chunk in enumerate(chunks, start=1):
            src = chunk.source
            loc = f"page {src.page}" if src.page else f"sheet {src.sheet}" if src.sheet else "source"
            parent_context = chunk.metadata.get("parent_context")
            if parent_context and parent_context != chunk.text:
                blocks.append(
                    f"[{idx}] {src.file_name} {loc} version={src.version_id}\n"
                    f"Matched chunk:\n{chunk.text}\n\nParent context:\n{parent_context}"
                )
            else:
                blocks.append(f"[{idx}] {src.file_name} {loc} version={src.version_id}\n{chunk.text}")
        return "\n\n".join(blocks)

    @staticmethod
    def _extract_location(query: str) -> str | None:
        if "Follow-up question:" in query:
            query = query.rsplit("Follow-up question:", 1)[1]
        cleaned = re.sub(
            r"\b(weather|temperature|forecast|today|now|current|currently|please)\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )
        match = re.search(r"(?:in|for|at)\s+([A-Za-z ,.-]+)", cleaned, flags=re.IGNORECASE)
        if match:
            location = re.sub(r"\s+", " ", match.group(1)).strip(" .,?!")
        else:
            location = re.sub(r"\s+", " ", cleaned).strip(" .,?!")
            if len(re.findall(r"[A-Za-z]+", location)) > 3:
                return None
        if not location:
            return None

        normalized = location.lower()
        if normalized in {"india", "in"}:
            return None
        aliases = {
            "banglore": "Bangalore,IN",
            "bangalore": "Bangalore,IN",
            "bengaluru": "Bangalore,IN",
            "bnglr": "Bangalore,IN",
            "blr": "Bangalore,IN",
            "delhi": "Delhi,IN",
            "new delhi": "New Delhi,IN",
            "mumbai": "Mumbai,IN",
            "kolkata": "Kolkata,IN",
            "chennai": "Chennai,IN",
            "hyderabad": "Hyderabad,IN",
            "hyd": "Hyderabad,IN",
            "pune": "Pune,IN",
        }
        return aliases.get(normalized, location)

    @staticmethod
    def _web_search_query(query: str) -> str:
        current = query
        if "Follow-up question:" in query:
            current = query.rsplit("Follow-up question:", 1)[1].splitlines()[0].strip()

        assistant_matches = re.findall(
            r"Assistant(?:\[[^\]]+\])?:\s*(.*?)(?=\n(?:User|Assistant)(?:\[[^\]]+\])?:|\nFollow-up question:|\Z)",
            query,
            flags=re.DOTALL,
        )
        previous_answer = re.sub(r"\s+", " ", assistant_matches[-1]).strip() if assistant_matches else ""

        current = re.sub(r"\b(look\s+up|search|look|check|browse|find|verify)\b", " ", current, flags=re.IGNORECASE)
        current = re.sub(r"\b(online|web|internet|for|up)\b", " ", current, flags=re.IGNORECASE)
        current = re.sub(r"\b(it|it'?s|its|this|that)\b", " ", current, flags=re.IGNORECASE)
        search_query = re.sub(r"\s+", " ", f"{current} {previous_answer}").strip(" .?!,")
        if not search_query:
            search_query = re.sub(r"\s+", " ", query).strip(" .?!,")
        return search_query[:400]
