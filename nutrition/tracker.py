"""Meal recording and daily nutrition tracking foundation.

Provides persistence and aggregation of meal nutrition records without external database dependencies.
Reuses the Phase 6 Step 1 nutrition calculator (nutrition.calculator) for all nutrient scaling.
"""

from __future__ import annotations

import csv
import uuid
from datetime import date as dt_date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from config.config import MEAL_LOG_PATH
from nutrition.calculator import calculate_nutrition

MEAL_LOG_COLUMNS = [
    "record_id",
    "date",
    "timestamp",
    "food_id",
    "food_name",
    "canonical_name",
    "quantity_g",
    "calories_kcal",
    "protein_g",
    "carbs_g",
    "fat_g",
    "fiber_g",
]


def _format_date(date_val: Optional[Union[str, dt_date, datetime]]) -> str:
    """Normalize input date to YYYY-MM-DD string format."""
    if date_val is None:
        return datetime.now().strftime("%Y-%m-%d")
    if isinstance(date_val, (dt_date, datetime)):
        return date_val.strftime("%Y-%m-%d")
    if isinstance(date_val, str):
        date_str = date_val.strip()
        # Validate format
        try:
            parsed = datetime.strptime(date_str, "%Y-%m-%d")
            return parsed.strftime("%Y-%m-%d")
        except ValueError as err:
            raise ValueError(
                f"Invalid date format '{date_val}'. Expected YYYY-MM-DD."
            ) from err
    raise TypeError(f"Unsupported date type: {type(date_val).__name__}")


def _format_timestamp(ts_val: Optional[Union[str, datetime]]) -> str:
    """Normalize timestamp to YYYY-MM-DD HH:MM:SS string format."""
    if ts_val is None:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(ts_val, datetime):
        return ts_val.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(ts_val, str):
        ts_str = ts_val.strip()
        # Accept either YYYY-MM-DD HH:MM:SS or ISO format
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                parsed = datetime.strptime(ts_str, fmt)
                return parsed.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
        raise ValueError(
            f"Invalid timestamp format '{ts_val}'. Expected YYYY-MM-DD HH:MM:SS."
        )
    raise TypeError(f"Unsupported timestamp type: {type(ts_val).__name__}")


def add_meal_record(
    food_identifier: str,
    quantity_g: Union[int, float],
    date: Optional[Union[str, dt_date, datetime]] = None,
    timestamp: Optional[Union[str, datetime]] = None,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Calculate scaled nutrition via Phase 6 Step 1 calculator and record the meal.

    Parameters
    ----------
    food_identifier : str
        Food canonical name (e.g. 'biryani') or food ID (e.g. 'FOOD_001').
    quantity_g : Union[int, float]
        Non-negative quantity in grams.
    date : Optional[Union[str, dt_date, datetime]]
        Calendar date for the record (default: current date in YYYY-MM-DD).
    timestamp : Optional[Union[str, datetime]]
        Timestamp for the record (default: current time in YYYY-MM-DD HH:MM:SS).
    log_path : Optional[Path]
        Optional custom CSV meal log path (default: config.MEAL_LOG_PATH).

    Returns
    -------
    Dict[str, Any]
        Result dictionary containing status, success flag, and the persisted meal record.
    """
    # 1. Reuse existing nutrition calculator for input validation and nutrient scaling
    nutrition = calculate_nutrition(food_identifier, quantity_g)

    # 2. Reject unresolved foods without creating a corrupt/fabricated persistent record
    if not nutrition["is_available"] or nutrition["nutrition_status"] == "unresolved":
        return {
            "success": False,
            "status": "unresolved",
            "message": nutrition.get(
                "message",
                f"Authoritative nutrition data is unavailable for '{food_identifier}'. Record was not created.",
            ),
            "food_name": nutrition.get("food_name"),
            "canonical_name": nutrition.get("canonical_name"),
            "quantity_g": float(quantity_g),
            "record": None,
        }

    # 3. Build Meal Record
    date_str = _format_date(date)
    ts_str = _format_timestamp(timestamp)
    record_id = f"meal_{uuid.uuid4().hex[:12]}"

    record: Dict[str, Any] = {
        "record_id": record_id,
        "date": date_str,
        "timestamp": ts_str,
        "food_id": nutrition["food_id"],
        "food_name": nutrition["food_name"],
        "canonical_name": nutrition["canonical_name"],
        "quantity_g": float(nutrition["quantity_g"]),
        "calories_kcal": float(nutrition["calories"]),
        "protein_g": float(nutrition["protein_g"]),
        "carbs_g": float(nutrition["carbs_g"]),
        "fat_g": float(nutrition["fat_g"]),
        "fiber_g": float(nutrition["fiber_g"]),
    }

    # 4. Append to CSV Meal Log (creates file with header if missing)
    target_path = Path(log_path) if log_path is not None else MEAL_LOG_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = target_path.exists() and target_path.stat().st_size > 0

    with open(target_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MEAL_LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

    return {
        "success": True,
        "status": "recorded",
        "message": f"Successfully recorded {record['quantity_g']}g of '{record['food_name']}'.",
        "record": record,
    }


def get_all_meal_records(log_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve all recorded meals from the persistent log.

    Parameters
    ----------
    log_path : Optional[Path]
        Optional custom CSV meal log path (default: config.MEAL_LOG_PATH).

    Returns
    -------
    List[Dict[str, Any]]
        List of parsed meal record dictionaries with numeric types converted to float.
    """
    target_path = Path(log_path) if log_path is not None else MEAL_LOG_PATH

    if not target_path.exists():
        return []

    records: List[Dict[str, Any]] = []

    with open(target_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {
                "record_id": row["record_id"],
                "date": row["date"],
                "timestamp": row["timestamp"],
                "food_id": row["food_id"],
                "food_name": row["food_name"],
                "canonical_name": row.get("canonical_name", ""),
                "quantity_g": float(row["quantity_g"]),
                "calories_kcal": float(row["calories_kcal"]),
                "protein_g": float(row["protein_g"]),
                "carbs_g": float(row["carbs_g"]),
                "fat_g": float(row["fat_g"]),
                "fiber_g": float(row["fiber_g"]),
            }
            records.append(parsed)

    return records


def get_meal_records_for_date(
    date: Union[str, dt_date, datetime],
    log_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Retrieve meal records for a specific calendar date (YYYY-MM-DD).

    Parameters
    ----------
    date : Union[str, dt_date, datetime]
        Calendar date to query.
    log_path : Optional[Path]
        Optional custom CSV meal log path.

    Returns
    -------
    List[Dict[str, Any]]
        List of meal records recorded on the specified date.
    """
    target_date = _format_date(date)
    all_records = get_all_meal_records(log_path=log_path)
    return [r for r in all_records if r["date"] == target_date]


def get_daily_nutrition_summary(
    date: Union[str, dt_date, datetime],
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Calculate aggregate nutrition totals from actual recorded meals for a selected date.

    Parameters
    ----------
    date : Union[str, dt_date, datetime]
        Target calendar date.
    log_path : Optional[Path]
        Optional custom CSV meal log path.

    Returns
    -------
    Dict[str, Any]
        Dictionary with date, meal count, summed calories, protein, carbs, fat, fiber,
        and the list of individual meal records.
    """
    target_date = _format_date(date)
    meals = get_meal_records_for_date(target_date, log_path=log_path)

    if not meals:
        return {
            "date": target_date,
            "meal_count": 0,
            "total_calories": 0.0,
            "total_protein_g": 0.0,
            "total_carbs_g": 0.0,
            "total_fat_g": 0.0,
            "total_fiber_g": 0.0,
            "meals": [],
        }

    total_calories = round(sum(m["calories_kcal"] for m in meals), 2)
    total_protein = round(sum(m["protein_g"] for m in meals), 2)
    total_carbs = round(sum(m["carbs_g"] for m in meals), 2)
    total_fat = round(sum(m["fat_g"] for m in meals), 2)
    total_fiber = round(sum(m["fiber_g"] for m in meals), 2)

    return {
        "date": target_date,
        "meal_count": len(meals),
        "total_calories": total_calories,
        "total_protein_g": total_protein,
        "total_carbs_g": total_carbs,
        "total_fat_g": total_fat,
        "total_fiber_g": total_fiber,
        "meals": meals,
    }

