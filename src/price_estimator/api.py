"""FastAPI application for the human-in-the-loop quote workflow.

Endpoints (all under /v1):
    POST /v1/quote              - Get an AI price estimate with bands and SHAP explanation
    POST /v1/quote/{id}/override - Override an AI quote with human price
    POST /v1/quote/{id}/send    - Mark a quote as sent to the customer
    POST /v1/quote/{id}/outcome - Record won/lost/expired outcome
    GET  /v1/quote/{id}         - Retrieve full quote detail with lifecycle
    GET  /v1/quotes             - List all quotes (summary view)
"""

import abc
import json
import logging
import re
import sqlite3
import time
import uuid
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel as PydanticModel
from pydantic import Field, field_validator

from price_estimator.data import (
    VALID_ESTIMATORS,
    VALID_MATERIALS,
    VALID_PROCESSES,
    VALID_QUANTITIES,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Price Estimator API",
    description=(
        "AI-assisted quoting for precision machined aerospace parts.\n\n"
        "**Materials:** Aluminum 6061, Aluminum 7075, Inconel 718, "
        "Stainless Steel 17-4 PH, Titanium Grade 5\n\n"
        "**Processes:** 3-Axis Milling, 5-Axis Milling, CNC Turning, "
        "Surface Grinding, Wire EDM\n\n"
        "**Estimators:** Sato-san, Suzuki-san, Tanaka-san\n\n"
        "**Quantities:** 1, 5, 10, 20, 50, 100"
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=86400,
)

router = APIRouter(prefix="/v1")


# ---------------------------------------------------------------------------
# Error envelope
# ---------------------------------------------------------------------------


class ErrorDetail(PydanticModel):
    """Structured error response body."""

    code: str
    message: str
    details: dict | None = None


class ErrorResponse(PydanticModel):
    """Wrapper for all error responses."""

    error: ErrorDetail


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Override FastAPI's default 422 to use our error envelope."""
    fields = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"] if part != "body")
        fields.append({"field": loc, "issue": err["msg"]})
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "UNPROCESSABLE_ENTITY",
                "message": f"{len(fields)} validation error(s) in request.",
                "details": {"fields": fields},
            }
        },
    )


def _error_response(status_code: int, code: str, message: str, details: dict | None = None):
    """Raise an HTTPException with our error envelope."""
    raise HTTPException(
        status_code=status_code,
        detail={"error": {"code": code, "message": message, "details": details}},
    )


# ---------------------------------------------------------------------------
# QuoteStore abstraction
# ---------------------------------------------------------------------------


class QuoteStore(abc.ABC):
    """Interface for persisting quotes and overrides."""

    @abc.abstractmethod
    def save_quote(
        self,
        quote_id: str,
        features: dict,
        model_price: float,
        range_low: float,
        range_high: float,
        aggressive_estimate: float | None,
        conservative_estimate: float | None,
        warnings: list[str],
        shap_explanation: list[dict] | None,
    ) -> None: ...

    @abc.abstractmethod
    def get_quote(self, quote_id: str) -> dict | None:
        """Return full quote dict or None."""

    @abc.abstractmethod
    def list_quotes(self) -> list[dict]:
        """Return list of quote summary dicts."""

    @abc.abstractmethod
    def save_override(
        self,
        override_id: str,
        quote_id: str,
        human_price: float,
        reason_category: str | None,
        reason_text: str | None,
        estimator_id: str | None,
        delta: float,
    ) -> None: ...

    @abc.abstractmethod
    def get_latest_override(self, quote_id: str) -> dict | None:
        """Return override dict or None."""

    @abc.abstractmethod
    def mark_sent(self, quote_id: str) -> None:
        """Mark a quote as sent to the customer."""

    @abc.abstractmethod
    def record_outcome(
        self,
        quote_id: str,
        outcome: str,
        reason: str | None,
        reason_text: str | None,
        final_negotiated_price: float | None,
        po_number: str | None,
    ) -> None: ...


class SqliteQuoteStore(QuoteStore):
    """SQLite-backed store for local development."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(self._db_path))

    def _ensure_tables(self) -> None:
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS quotes (
                quote_id TEXT PRIMARY KEY,
                features TEXT NOT NULL,
                model_price REAL NOT NULL,
                model_range_low REAL,
                model_range_high REAL,
                aggressive_estimate REAL,
                conservative_estimate REAL,
                warnings TEXT,
                shap_explanation TEXT,
                status TEXT NOT NULL DEFAULT 'draft',
                sent_at TEXT,
                outcome TEXT,
                outcome_reason TEXT,
                outcome_reason_text TEXT,
                outcome_negotiated_price REAL,
                outcome_po_number TEXT,
                outcome_at TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS overrides (
                override_id TEXT PRIMARY KEY,
                quote_id TEXT NOT NULL,
                human_price REAL NOT NULL,
                reason_category TEXT,
                reason_text TEXT,
                estimator_id TEXT,
                delta_from_model REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (quote_id) REFERENCES quotes(quote_id)
            )
        """)
        # Migrate existing tables that lack new columns
        try:
            conn.execute("SELECT status FROM quotes LIMIT 1")
        except sqlite3.OperationalError:
            for col, default in [
                ("aggressive_estimate REAL", None),
                ("conservative_estimate REAL", None),
                ("warnings TEXT", None),
                ("shap_explanation TEXT", None),
                ("status TEXT NOT NULL DEFAULT 'draft'", None),
                ("sent_at TEXT", None),
                ("outcome TEXT", None),
                ("outcome_reason TEXT", None),
                ("outcome_reason_text TEXT", None),
                ("outcome_negotiated_price REAL", None),
                ("outcome_po_number TEXT", None),
                ("outcome_at TEXT", None),
            ]:
                try:
                    conn.execute(f"ALTER TABLE quotes ADD COLUMN {col}")
                except sqlite3.OperationalError:
                    pass
            # Set existing rows to 'draft'
            conn.execute("UPDATE quotes SET status = 'draft' WHERE status IS NULL")
        conn.commit()
        conn.close()

    def save_quote(
        self,
        quote_id,
        features,
        model_price,
        range_low,
        range_high,
        aggressive_estimate,
        conservative_estimate,
        warnings,
        shap_explanation,
    ):
        conn = self._connect()
        conn.execute(
            "INSERT INTO quotes (quote_id, features, model_price, model_range_low, "
            "model_range_high, aggressive_estimate, conservative_estimate, warnings, "
            "shap_explanation, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?)",
            (
                quote_id,
                json.dumps(features),
                model_price,
                range_low,
                range_high,
                aggressive_estimate,
                conservative_estimate,
                json.dumps(warnings),
                json.dumps(shap_explanation),
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    def get_quote(self, quote_id):
        conn = self._connect()
        row = conn.execute(
            "SELECT quote_id, features, model_price, model_range_low, model_range_high, "
            "aggressive_estimate, conservative_estimate, warnings, shap_explanation, "
            "status, sent_at, outcome, outcome_reason, outcome_reason_text, "
            "outcome_negotiated_price, outcome_po_number, outcome_at, created_at "
            "FROM quotes WHERE quote_id = ?",
            (quote_id,),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return {
            "quote_id": row[0],
            "features": json.loads(row[1]),
            "model_price": row[2],
            "model_range_low": row[3],
            "model_range_high": row[4],
            "aggressive_estimate": row[5],
            "conservative_estimate": row[6],
            "warnings": json.loads(row[7]) if row[7] else [],
            "shap_explanation": json.loads(row[8]) if row[8] else None,
            "status": row[9] or "draft",
            "sent_at": row[10],
            "outcome": row[11],
            "outcome_reason": row[12],
            "outcome_reason_text": row[13],
            "outcome_negotiated_price": row[14],
            "outcome_po_number": row[15],
            "outcome_at": row[16],
            "created_at": row[17],
        }

    def list_quotes(self):
        conn = self._connect()
        rows = conn.execute(
            "SELECT quote_id, features, model_price, status, created_at "
            "FROM quotes ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        results = []
        for row in rows:
            features = json.loads(row[1])
            # Compute final price: check for override
            override = self.get_latest_override(row[0])
            final_price = override["human_price"] if override else row[2]
            results.append(
                {
                    "quote_id": row[0],
                    "status": row[3] or "draft",
                    "part_description": features.get("part_description", ""),
                    "material": features.get("material"),
                    "process": features.get("process"),
                    "quantity": features.get("quantity", 0),
                    "original_estimate": row[2],
                    "final_price": final_price,
                    "created_at": row[4],
                }
            )
        return results

    def save_override(
        self, override_id, quote_id, human_price, reason_category, reason_text, estimator_id, delta
    ):
        conn = self._connect()
        conn.execute(
            "INSERT INTO overrides (override_id, quote_id, human_price, reason_category, "
            "reason_text, estimator_id, delta_from_model, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                override_id,
                quote_id,
                human_price,
                reason_category,
                reason_text,
                estimator_id,
                delta,
                datetime.now().isoformat(),
            ),
        )
        # Move to review status if still in draft
        conn.execute(
            "UPDATE quotes SET status = 'review' WHERE quote_id = ? AND status = 'draft'",
            (quote_id,),
        )
        conn.commit()
        conn.close()

    def get_latest_override(self, quote_id):
        conn = self._connect()
        row = conn.execute(
            "SELECT human_price, reason_category, reason_text, estimator_id, "
            "delta_from_model, created_at FROM overrides WHERE quote_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (quote_id,),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return {
            "human_price": row[0],
            "reason_category": row[1],
            "reason_text": row[2],
            "estimator_id": row[3],
            "delta_from_model": row[4],
            "overridden_at": row[5],
        }

    def mark_sent(self, quote_id):
        conn = self._connect()
        conn.execute(
            "UPDATE quotes SET status = 'sent', sent_at = ? WHERE quote_id = ?",
            (datetime.now().isoformat(), quote_id),
        )
        conn.commit()
        conn.close()

    def record_outcome(
        self, quote_id, outcome, reason, reason_text, final_negotiated_price, po_number
    ):
        conn = self._connect()
        conn.execute(
            "UPDATE quotes SET status = ?, outcome = ?, outcome_reason = ?, "
            "outcome_reason_text = ?, outcome_negotiated_price = ?, "
            "outcome_po_number = ?, outcome_at = ? WHERE quote_id = ?",
            (
                outcome,
                outcome,
                reason,
                reason_text,
                final_negotiated_price,
                po_number,
                datetime.now().isoformat(),
                quote_id,
            ),
        )
        conn.commit()
        conn.close()


class DynamoQuoteStore(QuoteStore):
    """DynamoDB-backed store for Lambda deployment."""

    def __init__(self, quotes_table_name: str, overrides_table_name: str):
        import boto3

        dynamo = boto3.resource("dynamodb")
        self._quotes = dynamo.Table(quotes_table_name)
        self._overrides = dynamo.Table(overrides_table_name)

    def save_quote(
        self,
        quote_id,
        features,
        model_price,
        range_low,
        range_high,
        aggressive_estimate,
        conservative_estimate,
        warnings,
        shap_explanation,
    ):
        import calendar
        from decimal import Decimal

        now = datetime.now()
        ttl_epoch = calendar.timegm(now.timetuple()) + 90 * 86400  # 90-day TTL
        item = {
            "quote_id": quote_id,
            "features": json.dumps(features),
            "model_price": Decimal(str(round(model_price, 2))),
            "model_range_low": Decimal(str(round(range_low, 2))),
            "model_range_high": Decimal(str(round(range_high, 2))),
            "warnings": json.dumps(warnings),
            "shap_explanation": json.dumps(shap_explanation),
            "status": "draft",
            "created_at": now.isoformat(),
            "ttl": ttl_epoch,
        }
        if aggressive_estimate is not None:
            item["aggressive_estimate"] = Decimal(str(round(aggressive_estimate, 2)))
        if conservative_estimate is not None:
            item["conservative_estimate"] = Decimal(str(round(conservative_estimate, 2)))
        self._quotes.put_item(Item=item)

    def get_quote(self, quote_id):
        resp = self._quotes.get_item(Key={"quote_id": quote_id})
        item = resp.get("Item")
        if item is None:
            return None
        return {
            "quote_id": quote_id,
            "features": json.loads(item["features"]),
            "model_price": float(item["model_price"]),
            "model_range_low": float(item.get("model_range_low", item["model_price"])),
            "model_range_high": float(item.get("model_range_high", item["model_price"])),
            "aggressive_estimate": (
                float(item["aggressive_estimate"]) if "aggressive_estimate" in item else None
            ),
            "conservative_estimate": (
                float(item["conservative_estimate"]) if "conservative_estimate" in item else None
            ),
            "warnings": json.loads(item.get("warnings", "[]")),
            "shap_explanation": (
                json.loads(item["shap_explanation"]) if item.get("shap_explanation") else None
            ),
            "status": item.get("status", "draft"),
            "sent_at": item.get("sent_at"),
            "outcome": item.get("outcome"),
            "outcome_reason": item.get("outcome_reason"),
            "outcome_reason_text": item.get("outcome_reason_text"),
            "outcome_negotiated_price": (
                float(item["outcome_negotiated_price"])
                if item.get("outcome_negotiated_price")
                else None
            ),
            "outcome_po_number": item.get("outcome_po_number"),
            "outcome_at": item.get("outcome_at"),
            "created_at": item["created_at"],
        }

    def list_quotes(self):
        resp = self._quotes.scan(
            ProjectionExpression="quote_id, features, model_price, #s, created_at",
            ExpressionAttributeNames={"#s": "status"},
        )
        results = []
        for item in resp.get("Items", []):
            features = json.loads(item["features"])
            override = self.get_latest_override(item["quote_id"])
            final_price = override["human_price"] if override else float(item["model_price"])
            results.append(
                {
                    "quote_id": item["quote_id"],
                    "status": item.get("status", "draft"),
                    "part_description": features.get("part_description", ""),
                    "material": features.get("material"),
                    "process": features.get("process"),
                    "quantity": features.get("quantity", 0),
                    "original_estimate": float(item["model_price"]),
                    "final_price": final_price,
                    "created_at": item["created_at"],
                }
            )
        results.sort(key=lambda x: x["created_at"], reverse=True)
        return results

    def save_override(
        self, override_id, quote_id, human_price, reason_category, reason_text, estimator_id, delta
    ):
        from decimal import Decimal

        item = {
            "quote_id": quote_id,
            "override_id": override_id,
            "human_price": Decimal(str(round(human_price, 2))),
            "delta_from_model": Decimal(str(round(delta, 2))),
            "created_at": datetime.now().isoformat(),
        }
        if reason_category is not None:
            item["reason_category"] = reason_category
        if reason_text is not None:
            item["reason_text"] = reason_text
        if estimator_id is not None:
            item["estimator_id"] = estimator_id
        self._overrides.put_item(Item=item)
        # Move to review if still draft
        self._quotes.update_item(
            Key={"quote_id": quote_id},
            UpdateExpression="SET #s = :review",
            ConditionExpression="#s = :draft",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":review": "review", ":draft": "draft"},
        )

    def get_latest_override(self, quote_id):
        from boto3.dynamodb.conditions import Key

        resp = self._overrides.query(
            KeyConditionExpression=Key("quote_id").eq(quote_id),
            ScanIndexForward=False,
            Limit=1,
        )
        items = resp.get("Items", [])
        if not items:
            return None
        item = items[0]
        return {
            "human_price": float(item["human_price"]),
            "reason_category": item.get("reason_category"),
            "reason_text": item.get("reason_text"),
            "estimator_id": item.get("estimator_id"),
            "delta_from_model": float(item["delta_from_model"]),
            "overridden_at": item["created_at"],
        }

    def mark_sent(self, quote_id):
        self._quotes.update_item(
            Key={"quote_id": quote_id},
            UpdateExpression="SET #s = :sent, sent_at = :now",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":sent": "sent",
                ":now": datetime.now().isoformat(),
            },
        )

    def record_outcome(
        self, quote_id, outcome, reason, reason_text, final_negotiated_price, po_number
    ):
        from decimal import Decimal

        update_expr = "SET #s = :outcome, outcome = :outcome, outcome_at = :now"
        attr_values: dict = {
            ":outcome": outcome,
            ":now": datetime.now().isoformat(),
        }
        if reason is not None:
            update_expr += ", outcome_reason = :reason"
            attr_values[":reason"] = reason
        if reason_text is not None:
            update_expr += ", outcome_reason_text = :reason_text"
            attr_values[":reason_text"] = reason_text
        if final_negotiated_price is not None:
            update_expr += ", outcome_negotiated_price = :neg_price"
            attr_values[":neg_price"] = Decimal(str(round(final_negotiated_price, 2)))
        if po_number is not None:
            update_expr += ", outcome_po_number = :po"
            attr_values[":po"] = po_number
        self._quotes.update_item(
            Key={"quote_id": quote_id},
            UpdateExpression=update_expr,
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues=attr_values,
        )


# ---------------------------------------------------------------------------
# Store instance (set at startup)
# ---------------------------------------------------------------------------

DB_PATH = Path("outputs/overrides.db")
_store: QuoteStore | None = None


def get_store() -> QuoteStore:
    """Get the active QuoteStore, defaulting to SQLite for local dev."""
    global _store
    if _store is None:
        _store = SqliteQuoteStore(DB_PATH)
    return _store


def set_store(store: QuoteStore) -> None:
    """Set the active QuoteStore (called by lambda_handler or serve script)."""
    global _store
    _store = store


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class OverrideReasonCategory(StrEnum):
    """Structured override reason categories."""

    MATERIAL_HARDNESS = "material_hardness"
    GEOMETRY_COMPLEXITY = "geometry_complexity"
    SURFACE_FINISH = "surface_finish"
    TOOLING_DIFFICULTY = "tooling_difficulty"
    CUSTOMER_RELATIONSHIP = "customer_relationship"
    SCRAP_RISK = "scrap_risk"
    CERTIFICATION_REQUIREMENTS = "certification_requirements"
    OTHER = "other"


# Printable ASCII + common punctuation (no control chars, no emoji)
_PRINTABLE_RE = re.compile(r"^[\x20-\x7E]+$")


class QuoteRequest(PydanticModel):
    """Input for a new quote request."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "part_description": "Sensor Housing - threaded",
                    "material": "Inconel 718",
                    "process": "5-Axis Milling",
                    "quantity": 5,
                    "rush_job": False,
                    "lead_time_weeks": 4,
                    "estimator": "Sato-san",
                }
            ]
        }
    }

    part_description: str = Field(min_length=1, max_length=200)
    material: str | None = None
    process: str | None = None
    quantity: int = Field(gt=0)
    rush_job: bool = False
    lead_time_weeks: int = Field(ge=1, le=52)
    estimator: str | None = None

    @field_validator("part_description")
    @classmethod
    def validate_part_description(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("part_description must not be blank")
        if not _PRINTABLE_RE.match(v):
            raise ValueError("part_description must contain only printable ASCII characters")
        return v

    @field_validator("material")
    @classmethod
    def validate_material(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_MATERIALS:
            raise ValueError(
                f"Unknown material: '{v}'. Must be one of: {', '.join(VALID_MATERIALS)}"
            )
        return v

    @field_validator("process")
    @classmethod
    def validate_process(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_PROCESSES:
            raise ValueError(
                f"Unknown process: '{v}'. Must be one of: {', '.join(VALID_PROCESSES)}"
            )
        return v

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, v: int) -> int:
        if v not in VALID_QUANTITIES:
            raise ValueError(f"Invalid quantity: {v}. Must be one of: {VALID_QUANTITIES}")
        return v

    @field_validator("estimator")
    @classmethod
    def validate_estimator(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_ESTIMATORS:
            raise ValueError(
                f"Unknown estimator: '{v}'. Must be one of: {', '.join(VALID_ESTIMATORS)}"
            )
        return v


class QuoteResponse(PydanticModel):
    """Output from a quote request."""

    quote_id: str
    estimate: float
    aggressive_estimate: float | None = None
    conservative_estimate: float | None = None
    typical_range: dict | None = None
    warnings: list[str] = []
    shap_explanation: list[dict] | None = None


class OverrideRequest(PydanticModel):
    """Input for overriding a quote."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "human_price": 15500.00,
                    "reason_category": "material_hardness",
                    "reason_text": "Inconel work-hardens more than the model accounts for",
                    "estimator_id": "Tanaka-san",
                }
            ]
        }
    }

    human_price: float = Field(gt=0, le=10_000_000)
    reason_category: OverrideReasonCategory | None = None
    reason_text: str | None = Field(default=None, max_length=1000)
    estimator_id: str | None = None

    @field_validator("estimator_id")
    @classmethod
    def validate_estimator_id(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_ESTIMATORS:
            raise ValueError(
                f"Unknown estimator: '{v}'. Must be one of: {', '.join(VALID_ESTIMATORS)}"
            )
        return v


class OverrideResponse(PydanticModel):
    """Output from an override request."""

    stored: bool
    override_id: str
    delta_from_model: float
    delta_pct: float


class LossReasonCategory(StrEnum):
    """Structured loss reason categories."""

    PRICE_TOO_HIGH = "price_too_high"
    LEAD_TIME = "lead_time"
    WENT_WITH_COMPETITOR = "went_with_competitor"
    SCOPE_CHANGE = "scope_change"
    OTHER = "other"


class OutcomeRequest(PydanticModel):
    """Input for recording a quote outcome."""

    outcome: str = Field(pattern=r"^(won|lost|expired)$")
    reason: LossReasonCategory | None = None
    reason_text: str | None = Field(default=None, max_length=1000)
    final_negotiated_price: float | None = Field(default=None, gt=0)
    po_number: str | None = Field(default=None, max_length=100)


class QuoteSummary(PydanticModel):
    """Summary for list view."""

    quote_id: str
    status: str
    part_description: str
    material: str | None = None
    process: str | None = None
    quantity: int
    original_estimate: float
    final_price: float
    created_at: str


class QuoteDetail(PydanticModel):
    """Full quote detail with lifecycle."""

    quote_id: str
    status: str
    features: dict
    original_estimate: float
    aggressive_estimate: float | None = None
    conservative_estimate: float | None = None
    typical_range: dict | None = None
    warnings: list[str] = []
    shap_explanation: list[dict] | None = None
    override: dict | None = None
    outcome: dict | None = None
    final_price: float
    created_at: str
    sent_at: str | None = None
    timeline: list[dict] = []


# ---------------------------------------------------------------------------
# State (loaded at startup)
# ---------------------------------------------------------------------------

# These are set by the serve script after loading models
_models: dict = {}
_training_bounds = None
_prediction_bands: dict | None = None


def set_models(models: dict, training_bounds=None, prediction_bands=None):
    """Set the loaded models for the API to use.

    Called by the serve script after loading serialized models.

    Args:
        models: Dict mapping model name to trained model instance.
        training_bounds: TrainingBounds instance for OOD detection.
        prediction_bands: Dict mapping model name to empirical band dict.
    """
    global _models, _training_bounds, _prediction_bands
    _models = models
    _training_bounds = training_bounds
    _prediction_bands = prediction_bands


ESTIMATORS = ["Sato-san", "Suzuki-san", "Tanaka-san"]

# Models that don't use the estimator feature
DEBIASED_MODELS = {"M0", "M9"}


def _request_to_dataframe(req: QuoteRequest, estimator: str = "Suzuki-san") -> pd.DataFrame:
    """Convert a QuoteRequest to a single-row DataFrame.

    Args:
        req: The quote request.
        estimator: Estimator name to use. When the request doesn't specify
            an estimator, callers should use this to iterate over all
            estimators for range estimation.
    """
    return pd.DataFrame(
        [
            {
                "QuoteID": "API",
                "Date": pd.Timestamp.now(),
                "PartDescription": req.part_description,
                "Material": req.material,
                "Process": req.process,
                "Quantity": req.quantity,
                "LeadTimeWeeks": req.lead_time_weeks,
                "RushJob": req.rush_job,
                "Estimator": req.estimator or estimator,
                "TotalPrice_USD": 0.0,
            }
        ]
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/quote", response_model=QuoteResponse)
def create_quote(req: QuoteRequest) -> QuoteResponse:
    """Generate an AI price estimate for a new part quote."""
    _t0 = time.monotonic()
    if not _models:
        _error_response(503, "SERVICE_UNAVAILABLE", "No models loaded")

    estimator_provided = req.estimator is not None
    quote_id = f"Q-API-{uuid.uuid4().hex[:8]}"

    # Get predictions from all loaded models
    predictions = {}
    estimator_range = {}

    if estimator_provided:
        # Estimator known — straightforward prediction
        df = _request_to_dataframe(req)
        for name, model in _models.items():
            try:
                pred = model.predict(df)
                predictions[name] = float(pred[0])
            except Exception as e:
                logger.warning("Model %s failed: %s", name, e)
    else:
        # No estimator — use debiased models directly, run estimator-aware
        # models with all 3 estimators to get a range
        df = _request_to_dataframe(req)  # placeholder estimator for debiased
        per_est_preds: dict[str, list[float]] = {e: [] for e in ESTIMATORS}

        for name, model in _models.items():
            if name in DEBIASED_MODELS:
                try:
                    pred = model.predict(df)
                    predictions[name] = float(pred[0])
                except Exception as e:
                    logger.warning("Model %s failed: %s", name, e)
            else:
                est_preds = {}
                for est in ESTIMATORS:
                    df_est = _request_to_dataframe(req, estimator=est)
                    try:
                        pred = model.predict(df_est)
                        est_preds[est] = float(pred[0])
                        per_est_preds[est].append(float(pred[0]))
                    except Exception as e:
                        logger.warning("Model %s/%s failed: %s", name, est, e)
                if est_preds:
                    predictions[name] = float(np.median(list(est_preds.values())))

        for est in ESTIMATORS:
            if per_est_preds[est]:
                estimator_range[est] = float(np.median(per_est_preds[est]))

    if not predictions:
        logger.error(
            "All models failed",
            extra={"quote_id": quote_id, "part_description": req.part_description},
        )
        _error_response(500, "PREDICTION_FAILED", "All models failed to produce a prediction")

    # Recommendation from top-tier model consensus
    from price_estimator.predict import (
        apply_prediction_band,
        compute_recommendation,
        compute_shap_explanation,
        detect_ood,
    )

    # Use M2's empirical band (best model) for recommendation shifts
    band = _prediction_bands.get("M2") if _prediction_bands else None
    rec = compute_recommendation(predictions, band=band)

    estimate = rec["estimate"]
    aggressive_estimate = rec["win_bid"]
    conservative_estimate = rec["protect_margin"]

    # Typical range from empirical prediction band
    typical_range = None
    if band:
        low, high = apply_prediction_band(estimate, band)
        typical_range = {
            "low": round(low, 2),
            "high": round(high, 2),
            "coverage": band["coverage"],
        }

    # Warnings
    warnings = []
    if req.material is None:
        warnings.append("Missing material — estimate may be less accurate")
    if req.process is None:
        warnings.append("Missing process — estimate may be less accurate")

    if _training_bounds:
        ood = detect_ood(df, _training_bounds)
        if ood[0]["is_ood"]:
            warnings.extend(ood[0]["reasons"])

    if rec["family_divergence"] and rec["family_divergence"]["divergence_pct"] > 10:
        div = rec["family_divergence"]
        warnings.append(
            f"Linear/tree model divergence: {div['divergence_pct']:.0f}% "
            f"(linear ${div['linear_median']:,.0f} vs tree ${div['tree_median']:,.0f}) "
            f"— review recommended"
        )

    if rec["top_tier_spread_pct"] > 15:
        warnings.append(f"Top-tier model spread: {rec['top_tier_spread_pct']:.0f}%")

    if not estimator_provided:
        warnings.append("No estimator provided — using debiased consensus")

    # SHAP explanation (best-effort, tree models only)
    shap_explanation = None
    shap_candidates = ["M9", "M6", "M5"] if not estimator_provided else ["M6", "M7", "M5"]
    for name in shap_candidates:
        if name in _models:
            try:
                explanation = compute_shap_explanation(_models[name], df)
                shap_vals = explanation["shap_values"]
                if shap_vals.ndim > 1:
                    shap_vals = shap_vals[0]
                names = explanation["feature_names"]
                indices = np.argsort(np.abs(shap_vals))[::-1][:10]
                shap_explanation = [
                    {
                        "feature": names[idx],
                        "contribution": round(float(shap_vals[idx]), 2),
                    }
                    for idx in indices
                ]
            except Exception as e:
                logger.warning("SHAP computation failed for %s: %s", name, e)
            break

    # Store in database
    store = get_store()
    store.save_quote(
        quote_id=quote_id,
        features=req.model_dump(),
        model_price=estimate,
        range_low=typical_range["low"] if typical_range else estimate,
        range_high=typical_range["high"] if typical_range else estimate,
        aggressive_estimate=aggressive_estimate,
        conservative_estimate=conservative_estimate,
        warnings=warnings,
        shap_explanation=shap_explanation,
    )

    _latency_ms = (time.monotonic() - _t0) * 1000
    logger.info(
        "Quote created",
        extra={
            "quote_id": quote_id,
            "estimate": round(estimate, 2),
            "n_warnings": len(warnings),
            "latency_ms": round(_latency_ms, 1),
        },
    )

    return QuoteResponse(
        quote_id=quote_id,
        estimate=round(estimate, 2),
        aggressive_estimate=round(aggressive_estimate, 2) if aggressive_estimate else None,
        conservative_estimate=(round(conservative_estimate, 2) if conservative_estimate else None),
        typical_range=typical_range,
        warnings=warnings,
        shap_explanation=shap_explanation,
    )


@router.post("/quote/{quote_id}/override", response_model=OverrideResponse)
def override_quote(quote_id: str, req: OverrideRequest) -> OverrideResponse:
    """Override an AI quote with a human-determined price."""
    store = get_store()
    quote = store.get_quote(quote_id)

    if quote is None:
        _error_response(404, "QUOTE_NOT_FOUND", f"Quote {quote_id} not found")

    model_price = quote["model_price"]
    delta = req.human_price - model_price
    delta_pct = delta / model_price * 100

    override_id = f"OVR-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    store.save_override(
        override_id=override_id,
        quote_id=quote_id,
        human_price=req.human_price,
        reason_category=req.reason_category.value if req.reason_category else None,
        reason_text=req.reason_text,
        estimator_id=req.estimator_id,
        delta=delta,
    )

    logger.info(
        "Override recorded",
        extra={"quote_id": quote_id, "override_id": override_id, "delta_pct": round(delta_pct, 2)},
    )

    return OverrideResponse(
        stored=True,
        override_id=override_id,
        delta_from_model=round(delta, 2),
        delta_pct=round(delta_pct, 2),
    )


@router.get("/quote/{quote_id}", response_model=QuoteDetail)
def get_quote(quote_id: str) -> QuoteDetail:
    """Retrieve a quote with full lifecycle detail."""
    store = get_store()
    quote = store.get_quote(quote_id)

    if quote is None:
        _error_response(404, "QUOTE_NOT_FOUND", f"Quote {quote_id} not found")

    model_price = quote["model_price"]
    override = store.get_latest_override(quote_id)
    final_price = override["human_price"] if override else model_price

    # Build typical range from stored bounds
    typical_range = None
    if quote.get("model_range_low") and quote.get("model_range_high"):
        low, high = quote["model_range_low"], quote["model_range_high"]
        if low != high:
            typical_range = {"low": round(low, 2), "high": round(high, 2), "coverage": 0.8}

    # Build outcome dict
    outcome_dict = None
    if quote.get("outcome"):
        outcome_dict = {
            "result": quote["outcome"],
            "reason": quote.get("outcome_reason"),
            "reason_text": quote.get("outcome_reason_text"),
            "final_negotiated_price": quote.get("outcome_negotiated_price"),
            "po_number": quote.get("outcome_po_number"),
            "recorded_at": quote.get("outcome_at"),
        }

    # Build timeline
    timeline = [
        {
            "event": "Created",
            "timestamp": quote["created_at"],
            "detail": f"AI estimate ${model_price:,.2f}",
        },
    ]
    if override:
        timeline.append(
            {
                "event": "Overridden",
                "timestamp": override["overridden_at"],
                "detail": (
                    f"to ${override['human_price']:,.2f}"
                    + (
                        f" ({override['reason_category']})"
                        if override.get("reason_category")
                        else ""
                    )
                ),
            }
        )
    if quote.get("sent_at"):
        timeline.append(
            {
                "event": "Sent",
                "timestamp": quote["sent_at"],
                "detail": f"at ${final_price:,.2f}",
            }
        )
    if quote.get("outcome_at"):
        timeline.append(
            {
                "event": quote["outcome"].capitalize() if quote.get("outcome") else "Outcome",
                "timestamp": quote["outcome_at"],
                "detail": quote.get("outcome_reason"),
            }
        )

    return QuoteDetail(
        quote_id=quote_id,
        status=quote.get("status", "draft"),
        features=quote.get("features", {}),
        original_estimate=round(model_price, 2),
        aggressive_estimate=quote.get("aggressive_estimate"),
        conservative_estimate=quote.get("conservative_estimate"),
        typical_range=typical_range,
        warnings=quote.get("warnings", []),
        shap_explanation=quote.get("shap_explanation"),
        override=override,
        outcome=outcome_dict,
        final_price=round(final_price, 2),
        created_at=quote.get("created_at", ""),
        sent_at=quote.get("sent_at"),
        timeline=timeline,
    )


@router.get("/quotes", response_model=list[QuoteSummary])
def list_quotes() -> list[QuoteSummary]:
    """List all quotes (summary view)."""
    store = get_store()
    return [QuoteSummary(**q) for q in store.list_quotes()]


@router.post("/quote/{quote_id}/send")
def send_quote(quote_id: str) -> dict:
    """Mark a quote as sent to the customer."""
    store = get_store()
    quote = store.get_quote(quote_id)
    if quote is None:
        _error_response(404, "QUOTE_NOT_FOUND", f"Quote {quote_id} not found")

    if quote.get("status") == "sent":
        _error_response(409, "ALREADY_SENT", f"Quote {quote_id} is already sent")

    if quote.get("status") in ("won", "lost", "expired"):
        _error_response(
            409,
            "QUOTE_CLOSED",
            f"Quote {quote_id} is already closed ({quote['status']})",
        )

    store.mark_sent(quote_id)
    logger.info("Quote sent", extra={"quote_id": quote_id})
    return {"ok": True}


@router.post("/quote/{quote_id}/outcome")
def record_outcome(quote_id: str, req: OutcomeRequest) -> dict:
    """Record the outcome (won/lost/expired) of a sent quote."""
    store = get_store()
    quote = store.get_quote(quote_id)
    if quote is None:
        _error_response(404, "QUOTE_NOT_FOUND", f"Quote {quote_id} not found")

    if quote.get("status") != "sent":
        _error_response(
            409,
            "INVALID_STATE",
            f"Quote {quote_id} is in '{quote.get('status')}' state, "
            "must be 'sent' to record outcome",
        )

    store.record_outcome(
        quote_id=quote_id,
        outcome=req.outcome,
        reason=req.reason.value if req.reason else None,
        reason_text=req.reason_text,
        final_negotiated_price=req.final_negotiated_price,
        po_number=req.po_number,
    )
    logger.info("Outcome recorded", extra={"quote_id": quote_id, "outcome": req.outcome})
    return {"ok": True}


# Register router after all routes are defined
app.include_router(router)
