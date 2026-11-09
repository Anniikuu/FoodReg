"""Deterministic, data-driven nutrition recommendation foundation.

Generates simple, explainable, non-clinical guidance by comparing daily calorie intake
against the user's daily calorie target, reporting current protein intake, and offering
simple nutrition-aware food suggestions based ONLY on verified foods in the nutrition database.
Reuses existing calorie comparison logic without external APIs or LLMs.
"""

from __future__ import annotations

import math
from datetime import date as dt_date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nutrition.calculator import load_nutrition_database
from nutrition.targets import compute_calorie_comparison, load_user_profile
from nutrition.tracker import get_daily_nutrition_summary

# Fixed threshold defining when intake is "close" to target (within 150 kcal)
CLOSE_TO_TARGET_THRESHOLD_KCAL = 150.0

# Maximum number of food suggestions returned
MAX_FOOD_SUGGESTIONS = 3
MAX_SUGGESTIONS = MAX_FOOD_SUGGESTIONS


def get_food_suggestions(
    remaining_calories: float,
    csv_path: Optional[Path] = None,
    max_suggestions: int = MAX_FOOD_SUGGESTIONS,
) -> List[Dict[str, Any]]:
    """Select a deterministic ranked list of verified food suggestions.

    Considers ONLY foods present in the nutrition database with 'verified' status.
    Foods with unresolved status or missing nutrient values are never selected.
    Candidate foods must have calories per 100g not exceeding the remaining calories.
    Ranked deterministically by protein content (highest first), then lower calories,
    with alphabetical canonical name as a tie-breaker.

    Parameters
    ----------
    remaining_calories : float
        Remaining calories under the daily calorie target.
    csv_path : Optional[Path]
        Optional custom path to nutrition_database.csv.
    max_suggestions : int
        Maximum number of food suggestions to return (default: 3).

    Returns
    -------
    List[Dict[str, Any]]
        List of suggested food records (at most max_suggestions).
    """
    if remaining_calories is None:
        return []

    try:
        rem_cal = float(remaining_calories)
    except (TypeError, ValueError):
        return []

    if math.isnan(rem_cal) or math.isinf(rem_cal) or rem_cal <= 0.0:
        return []

    db = load_nutrition_database(csv_path=csv_path)

    # Deduplicate records by unique food_id
    seen_ids = set()
    unique_records: List[Dict[str, Any]] = []
    for record in db.values():
        fid = record.get("food_id")
        if fid and fid not in seen_ids:
            seen_ids.add(fid)
            unique_records.append(record)

    # Filter: verified foods only, valid numbers, calories per 100g <= remaining_calories
    candidates: List[Dict[str, Any]] = []
    for rec in unique_records:
        status = rec.get("nutrition_status", "").strip().lower()
        if status != "verified":
            continue

        cal_100g = rec.get("calories_kcal_100g")
        prot_100g = rec.get("protein_g_100g")

        if cal_100g is None or prot_100g is None:
            continue

        try:
            cal_val = float(cal_100g)
            prot_val = float(prot_100g)
        except (TypeError, ValueError):
            continue

        if math.isnan(cal_val) or math.isnan(prot_val) or cal_val <= 0.0:
            continue

        if cal_val <= rem_cal:
            candidates.append({
                "food_id": rec.get("food_id", ""),
                "food_name": rec.get("food_name", ""),
                "canonical_name": rec.get("canonical_name", ""),
                "calories_kcal_100g": cal_val,
                "protein_g_100g": prot_val,
            })

    # Sort deterministically:
    # 1. Higher protein first (-protein_g_100g)
    # 2. Lower calories (calories_kcal_100g)
    # 3. Alphabetical tie-breaker (canonical_name)
    candidates.sort(
        key=lambda r: (-r["protein_g_100g"], r["calories_kcal_100g"], r["canonical_name"])
    )

    cap = max(0, int(max_suggestions))
    return candidates[:cap]


def generate_nutrition_recommendation(
    consumed_calories: float,
    daily_target_calories: Optional[float],
    protein_g: float,
    meal_count: int = 0,
    close_threshold_kcal: float = CLOSE_TO_TARGET_THRESHOLD_KCAL,
    csv_path: Optional[Path] = None,
    include_suggestions: bool = True,
) -> Dict[str, Any]:
    """Generate deterministic nutrition guidance and optional food suggestions.

    Parameters
    ----------
    consumed_calories : float
        Total calories consumed for the selected date.
    daily_target_calories : Optional[float]
        Daily calorie target from user profile, or None if no profile exists.
    protein_g : float
        Actual protein consumed in grams for the selected date.
    meal_count : int
        Number of meals recorded for the selected date (default: 0).
    close_threshold_kcal : float
        Calorie margin within which intake is considered 'close to target' (default: 150.0 kcal).
    csv_path : Optional[Path]
        Optional custom path to nutrition_database.csv.
    include_suggestions : bool
        Whether to generate food suggestions when below target (default: True).

    Returns
    -------
    Dict[str, Any]
        Dictionary with status, title, message, protein_guidance, suggestions, and stats.
    """
    consumed_val = round(float(consumed_calories), 2)
    protein_val = round(float(protein_g), 2)
    protein_guidance = f"Current protein intake: {protein_val:.2f} g."

    # Case 1: Missing Profile (no target configured)
    if daily_target_calories is None:
        return {
            "status": "no_profile",
            "title": "Profile Required",
            "message": (
                "A user profile and daily calorie target are required for personalized calorie guidance. "
                "Please configure your profile first."
            ),
            "protein_guidance": protein_guidance,
            "consumed_calories": consumed_val,
            "target_calories": None,
            "remaining_calories": None,
            "protein_g": protein_val,
            "meal_count": meal_count,
            "suggestions": [],
            "suggestions_explanation": "Profile not configured; food suggestions unavailable.",
        }

    target_val = round(float(daily_target_calories), 2)
    remaining_val = round(target_val - consumed_val, 2)

    # Case 2: No meals recorded for this date
    if meal_count == 0 and consumed_val == 0.0:
        return {
            "status": "no_meals",
            "title": "No Meals Recorded",
            "message": (
                "No meals have been recorded for this date yet. "
                "Record your meals to receive personalized intake guidance."
            ),
            "protein_guidance": protein_guidance,
            "consumed_calories": 0.0,
            "target_calories": target_val,
            "remaining_calories": target_val,
            "protein_g": 0.0,
            "meal_count": 0,
            "suggestions": [],
            "suggestions_explanation": "No meals recorded for this date; suggestions omitted.",
        }

    # Case 3: Target exceeded (remaining < 0)
    if remaining_val < 0:
        return {
            "status": "target_exceeded",
            "title": "Target Exceeded",
            "message": (
                "Your daily calorie target has been exceeded. "
                "Consider keeping remaining meals lighter and nutrient-balanced."
            ),
            "protein_guidance": protein_guidance,
            "consumed_calories": consumed_val,
            "target_calories": target_val,
            "remaining_calories": remaining_val,
            "protein_g": protein_val,
            "meal_count": meal_count,
            "suggestions": [],
            "suggestions_explanation": "Daily calorie target exceeded; additional food suggestions omitted.",
        }

    # Case 4: Target reached exactly (remaining == 0)
    if remaining_val == 0.0:
        return {
            "status": "target_met",
            "title": "Target Reached",
            "message": (
                "You have reached your daily calorie target. "
                "Avoid adding unnecessary calories unless required by your plan."
            ),
            "protein_guidance": protein_guidance,
            "consumed_calories": consumed_val,
            "target_calories": target_val,
            "remaining_calories": 0.0,
            "protein_g": protein_val,
            "meal_count": meal_count,
            "suggestions": [],
            "suggestions_explanation": "Daily calorie target reached; additional food suggestions omitted.",
        }

    # Case 5: Calories close to target (0 < remaining <= close_threshold_kcal)
    if 0 < remaining_val <= close_threshold_kcal:
        return {
            "status": "close_to_target",
            "title": "Intake Close to Target",
            "message": (
                "Your calorie intake is close to your daily target. "
                "Continue with balanced meals and monitor your remaining intake."
            ),
            "protein_guidance": protein_guidance,
            "consumed_calories": consumed_val,
            "target_calories": target_val,
            "remaining_calories": remaining_val,
            "protein_g": protein_val,
            "meal_count": meal_count,
            "suggestions": [],
            "suggestions_explanation": (
                "Calorie intake is close to target; additional food suggestions omitted to avoid exceeding target."
            ),
        }

    # Case 6: Calories significantly below target (remaining > close_threshold_kcal)
    suggestions: List[Dict[str, Any]] = []
    explanation: str = ""
    if include_suggestions:
        try:
            suggestions = get_food_suggestions(
                remaining_calories=remaining_val,
                csv_path=csv_path,
                max_suggestions=MAX_FOOD_SUGGESTIONS,
            )
            if suggestions:
                explanation = (
                    f"Top {len(suggestions)} verified food options with valuable protein "
                    f"fitting within your remaining {remaining_val:.1f} kcal."
                )
            else:
                explanation = (
                    "No verified food options in the database fit within your remaining calorie allowance."
                )
        except Exception as e:
            suggestions = []
            explanation = f"Could not load food suggestions: {e}"
    else:
        explanation = "Food suggestions disabled."

    return {
        "status": "below_target",
        "title": "Intake Below Target",
        "message": (
            "Your calorie intake is currently below your daily target. "
            "Consider adding a balanced meal or snack."
        ),
        "protein_guidance": protein_guidance,
        "consumed_calories": consumed_val,
        "target_calories": target_val,
        "remaining_calories": remaining_val,
        "protein_g": protein_val,
        "meal_count": meal_count,
        "suggestions": suggestions,
        "suggestions_explanation": explanation,
    }


def get_daily_recommendation(
    date_val: Union[str, dt_date, datetime],
    profile_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    close_threshold_kcal: float = CLOSE_TO_TARGET_THRESHOLD_KCAL,
    include_suggestions: bool = True,
) -> Dict[str, Any]:
    """Retrieve date intake and user profile to generate deterministic guidance.

    Parameters
    ----------
    date_val : Union[str, dt_date, datetime]
        Target calendar date.
    profile_path : Optional[Path]
        Optional custom user profile JSON path.
    log_path : Optional[Path]
        Optional custom meal log CSV path.
    csv_path : Optional[Path]
        Optional custom path to nutrition_database.csv.
    close_threshold_kcal : float
        Calorie margin for 'close to target' status.
    include_suggestions : bool
        Whether to generate food suggestions when below target.

    Returns
    -------
    Dict[str, Any]
        Complete recommendation dictionary including target, consumed, date, and suggestions.
    """
    summary = get_daily_nutrition_summary(date_val, log_path=log_path)
    profile_payload = load_user_profile(profile_path=profile_path)

    daily_target = (
        float(profile_payload["targets"]["daily_calorie_target_kcal"])
        if profile_payload and "targets" in profile_payload
        else None
    )

    rec = generate_nutrition_recommendation(
        consumed_calories=summary["total_calories"],
        daily_target_calories=daily_target,
        protein_g=summary["total_protein_g"],
        meal_count=summary["meal_count"],
        close_threshold_kcal=close_threshold_kcal,
        csv_path=csv_path,
        include_suggestions=include_suggestions,
    )
    rec["date"] = summary["date"]
    return rec
