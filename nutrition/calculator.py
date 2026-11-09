"""Core nutrition calculation foundation for scaled nutrient retrieval.

Uses the standard per-100g formula:
    nutrient_for_quantity = nutrient_per_100g * quantity_g / 100

Validates inputs, handles verified and unresolved foods safely, and avoids
fabricating missing nutrition values.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Centralized config
from config.config import NUTRITION_DATABASE_PATH

# Cached database records
_DATABASE_CACHE: Optional[Dict[str, Dict[str, Any]]] = None

NUTRIENT_FIELDS = [
    "calories_kcal_100g",
    "protein_g_100g",
    "carbs_g_100g",
    "fat_g_100g",
    "fiber_g_100g",
]


def load_nutrition_database(
    csv_path: Optional[Path] = None,
    reload: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Load and index nutrition database from CSV.

    Parameters
    ----------
    csv_path : Optional[Path]
        Path to nutrition_database.csv. Defaults to config.NUTRITION_DATABASE_PATH.
    reload : bool
        If True, reloads from disk bypassing in-memory cache.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping lowercased canonical names and food IDs to food records.
    """
    global _DATABASE_CACHE

    if _DATABASE_CACHE is not None and not reload and csv_path is None:
        return _DATABASE_CACHE

    target_path = Path(csv_path) if csv_path is not None else NUTRITION_DATABASE_PATH

    if not target_path.exists():
        raise FileNotFoundError(f"Nutrition database not found at {target_path}")

    records: Dict[str, Dict[str, Any]] = {}

    with open(target_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row.get("nutrition_status", "").strip().lower()
            canonical_name = row.get("canonical_name", "").strip().lower()
            food_id = row.get("food_id", "").strip().upper()

            parsed_nutrients: Dict[str, Optional[float]] = {}
            for field in NUTRIENT_FIELDS:
                raw_val = row.get(field, "").strip()
                if raw_val:
                    parsed_nutrients[field] = float(raw_val)
                else:
                    parsed_nutrients[field] = None

            record: Dict[str, Any] = {
                "food_id": row.get("food_id", "").strip(),
                "food_name": row.get("food_name", "").strip(),
                "canonical_name": canonical_name,
                "food_category": row.get("food_category", "").strip(),
                "preparation": row.get("preparation", "").strip(),
                "nutrition_status": status,
                "calories_kcal_100g": parsed_nutrients["calories_kcal_100g"],
                "protein_g_100g": parsed_nutrients["protein_g_100g"],
                "carbs_g_100g": parsed_nutrients["carbs_g_100g"],
                "fat_g_100g": parsed_nutrients["fat_g_100g"],
                "fiber_g_100g": parsed_nutrients["fiber_g_100g"],
                "source": row.get("source", "").strip(),
                "source_food_id": row.get("source_food_id", "").strip(),
            }

            # Index by both canonical_name and food_id
            if canonical_name:
                records[canonical_name] = record
            if food_id:
                records[food_id.lower()] = record

    if csv_path is None:
        _DATABASE_CACHE = records

    return records


def get_food_nutrition_record(
    food_identifier: str,
    csv_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Look up a raw nutrition record by canonical name or food ID.

    Parameters
    ----------
    food_identifier : str
        Food canonical name (e.g., 'biryani') or food ID (e.g., 'FOOD_001').
    csv_path : Optional[Path]
        Optional custom path to nutrition_database.csv.

    Returns
    -------
    Dict[str, Any]
        Raw food record dictionary.
    """
    if not isinstance(food_identifier, str):
        raise TypeError(f"food_identifier must be a string, got {type(food_identifier).__name__}")

    key = food_identifier.strip().lower()
    db = load_nutrition_database(csv_path=csv_path)

    if key not in db:
        raise KeyError(f"Food '{food_identifier}' not found in nutrition database")

    return db[key]


def calculate_nutrition(
    food_identifier: str,
    quantity_g: Union[int, float],
    csv_path: Optional[Path] = None,
    decimal_places: int = 2,
) -> Dict[str, Any]:
    """Calculate scaled nutrition values for a specific food and quantity in grams.

    Formula:
        nutrient_scaled = nutrient_per_100g * (quantity_g / 100.0)

    Parameters
    ----------
    food_identifier : str
        Food canonical name (e.g., 'biryani', 'idli') or food ID (e.g., 'FOOD_001').
    quantity_g : Union[int, float]
        Non-negative numeric quantity in grams.
    csv_path : Optional[Path]
        Optional custom path to nutrition_database.csv.
    decimal_places : int
        Number of decimal places for final rounded values (default: 2).

    Returns
    -------
    Dict[str, Any]
        Calculated nutrition results containing status, scaled values, and provenance.
    """
    # 1. Input Validation - Quantity Type
    if isinstance(quantity_g, bool):
        raise TypeError("Quantity must be a numeric value (int or float), not boolean")

    if not isinstance(quantity_g, (int, float)):
        raise TypeError(
            f"Quantity must be a numeric value in grams, got {type(quantity_g).__name__}"
        )

    # 2. Input Validation - Quantity Numeric Validity
    if math.isnan(quantity_g) or math.isinf(quantity_g):
        raise ValueError("Quantity in grams must be a finite number")

    if quantity_g < 0:
        raise ValueError(f"Quantity in grams cannot be negative, got {quantity_g}")

    # 3. Lookup Food Record
    record = get_food_nutrition_record(food_identifier, csv_path=csv_path)

    # 4. Handle Unresolved Foods (strictly no fabrication or zero-filling)
    if record["nutrition_status"] == "unresolved":
        return {
            "food_id": record["food_id"],
            "food_name": record["food_name"],
            "canonical_name": record["canonical_name"],
            "quantity_g": float(quantity_g),
            "is_available": False,
            "nutrition_status": "unresolved",
            "message": (
                f"Authoritative nutrition data is currently unavailable for "
                f"'{record['food_name']}'. Nutrient values are not estimated."
            ),
            "calories": None,
            "protein_g": None,
            "carbs_g": None,
            "fat_g": None,
            "fiber_g": None,
            "source": record["source"],
            "source_food_id": record["source_food_id"],
        }

    # 5. Handle Zero Grams (for verified foods)
    if quantity_g == 0:
        return {
            "food_id": record["food_id"],
            "food_name": record["food_name"],
            "canonical_name": record["canonical_name"],
            "quantity_g": 0.0,
            "is_available": True,
            "nutrition_status": "verified",
            "calories": 0.0,
            "protein_g": 0.0,
            "carbs_g": 0.0,
            "fat_g": 0.0,
            "fiber_g": 0.0,
            "per_100g": {
                "calories": record["calories_kcal_100g"],
                "protein_g": record["protein_g_100g"],
                "carbs_g": record["carbs_g_100g"],
                "fat_g": record["fat_g_100g"],
                "fiber_g": record["fiber_g_100g"],
            },
            "source": record["source"],
            "source_food_id": record["source_food_id"],
        }

    # 6. Standard per-100g calculation
    factor = float(quantity_g) / 100.0

    calories = round(record["calories_kcal_100g"] * factor, decimal_places)
    protein = round(record["protein_g_100g"] * factor, decimal_places)
    carbs = round(record["carbs_g_100g"] * factor, decimal_places)
    fat = round(record["fat_g_100g"] * factor, decimal_places)
    fiber = round(record["fiber_g_100g"] * factor, decimal_places)

    return {
        "food_id": record["food_id"],
        "food_name": record["food_name"],
        "canonical_name": record["canonical_name"],
        "quantity_g": float(quantity_g),
        "is_available": True,
        "nutrition_status": "verified",
        "calories": calories,
        "protein_g": protein,
        "carbs_g": carbs,
        "fat_g": fat,
        "fiber_g": fiber,
        "per_100g": {
            "calories": record["calories_kcal_100g"],
            "protein_g": record["protein_g_100g"],
            "carbs_g": record["carbs_g_100g"],
            "fat_g": record["fat_g_100g"],
            "fiber_g": record["fiber_g_100g"],
        },
        "source": record["source"],
        "source_food_id": record["source_food_id"],
    }

