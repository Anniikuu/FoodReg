"""Deterministic user nutrition targets and daily calorie calculation foundation.

Calculates Basal Metabolic Rate (BMR) via the Mifflin-St Jeor equation,
Total Daily Energy Expenditure (TDEE) via standardized activity multipliers,
and Daily Calorie Targets adjusted for weight management goals.
"""

from __future__ import annotations

import json
import math
from datetime import date as dt_date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from config.config import USER_PROFILE_PATH
from nutrition.tracker import get_daily_nutrition_summary

# Standard activity level multipliers
ACTIVITY_MULTIPLIERS: Dict[str, float] = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.9,
}

# Goal-based calorie adjustments (academic prototype rules)
GOAL_ADJUSTMENTS: Dict[str, float] = {
    "lose_weight": -500.0,
    "maintain_weight": 0.0,
    "gain_weight": 300.0,
}

SUPPORTED_SEXES = {"male", "female"}


def validate_profile_inputs(
    age_years: Union[int, float],
    sex: str,
    height_cm: Union[int, float],
    weight_kg: Union[int, float],
    activity_level: str,
    goal: str,
) -> Dict[str, Any]:
    """Validate user profile inputs and return normalized values.

    Parameters
    ----------
    age_years : Union[int, float]
        Positive integer-like age in years.
    sex : str
        Biological sex: 'male' or 'female'.
    height_cm : Union[int, float]
        Positive height in centimeters.
    weight_kg : Union[int, float]
        Positive weight in kilograms.
    activity_level : str
        One of: 'sedentary', 'light', 'moderate', 'active', 'very_active'.
    goal : str
        One of: 'lose_weight', 'maintain_weight', 'gain_weight'.

    Returns
    -------
    Dict[str, Any]
        Dictionary with strictly validated and normalized fields.
    """
    # 1. Age validation
    if isinstance(age_years, bool):
        raise TypeError("age_years must be a numeric value, not boolean")
    if not isinstance(age_years, (int, float)):
        raise TypeError(f"age_years must be numeric, got {type(age_years).__name__}")
    if math.isnan(age_years) or math.isinf(age_years):
        raise ValueError("age_years must be a finite number")
    if age_years <= 0:
        raise ValueError(f"age_years must be greater than 0, got {age_years}")

    # 2. Sex validation
    if not isinstance(sex, str):
        raise TypeError(f"sex must be a string, got {type(sex).__name__}")
    norm_sex = sex.strip().lower()
    if norm_sex not in SUPPORTED_SEXES:
        raise ValueError(
            f"Unsupported sex '{sex}'. Supported values: {sorted(SUPPORTED_SEXES)}"
        )

    # 3. Height validation
    if isinstance(height_cm, bool):
        raise TypeError("height_cm must be a numeric value, not boolean")
    if not isinstance(height_cm, (int, float)):
        raise TypeError(f"height_cm must be numeric, got {type(height_cm).__name__}")
    if math.isnan(height_cm) or math.isinf(height_cm):
        raise ValueError("height_cm must be a finite number")
    if height_cm <= 0:
        raise ValueError(f"height_cm must be greater than 0, got {height_cm}")

    # 4. Weight validation
    if isinstance(weight_kg, bool):
        raise TypeError("weight_kg must be a numeric value, not boolean")
    if not isinstance(weight_kg, (int, float)):
        raise TypeError(f"weight_kg must be numeric, got {type(weight_kg).__name__}")
    if math.isnan(weight_kg) or math.isinf(weight_kg):
        raise ValueError("weight_kg must be a finite number")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be greater than 0, got {weight_kg}")

    # 5. Activity level validation
    if not isinstance(activity_level, str):
        raise TypeError(f"activity_level must be a string, got {type(activity_level).__name__}")
    norm_activity = activity_level.strip().lower()
    if norm_activity not in ACTIVITY_MULTIPLIERS:
        raise ValueError(
            f"Unsupported activity_level '{activity_level}'. "
            f"Supported values: {sorted(ACTIVITY_MULTIPLIERS.keys())}"
        )

    # 6. Goal validation
    if not isinstance(goal, str):
        raise TypeError(f"goal must be a string, got {type(goal).__name__}")
    norm_goal = goal.strip().lower()
    if norm_goal not in GOAL_ADJUSTMENTS:
        raise ValueError(
            f"Unsupported goal '{goal}'. Supported values: {sorted(GOAL_ADJUSTMENTS.keys())}"
        )

    return {
        "age_years": int(age_years),
        "sex": norm_sex,
        "height_cm": float(height_cm),
        "weight_kg": float(weight_kg),
        "activity_level": norm_activity,
        "goal": norm_goal,
    }


def calculate_bmr(
    weight_kg: float,
    height_cm: float,
    age_years: int,
    sex: str,
) -> float:
    """Calculate Basal Metabolic Rate using the Mifflin-St Jeor equation.

    Male:   BMR = (10 * weight_kg) + (6.25 * height_cm) - (5 * age_years) + 5
    Female: BMR = (10 * weight_kg) + (6.25 * height_cm) - (5 * age_years) - 161
    """
    base = (10.0 * weight_kg) + (6.25 * height_cm) - (5.0 * age_years)
    if sex == "male":
        return base + 5.0
    elif sex == "female":
        return base - 161.0
    else:
        raise ValueError(f"Unsupported sex '{sex}' for BMR calculation")


def calculate_user_targets(
    age_years: Union[int, float],
    sex: str,
    height_cm: Union[int, float],
    weight_kg: Union[int, float],
    activity_level: str,
    goal: str,
    decimal_places: int = 2,
) -> Dict[str, Any]:
    """Calculate deterministic BMR, TDEE, and daily calorie target.

    Parameters
    ----------
    age_years : Union[int, float]
        Age in years (> 0).
    sex : str
        'male' or 'female'.
    height_cm : Union[int, float]
        Height in cm (> 0).
    weight_kg : Union[int, float]
        Weight in kg (> 0).
    activity_level : str
        'sedentary', 'light', 'moderate', 'active', or 'very_active'.
    goal : str
        'lose_weight', 'maintain_weight', or 'gain_weight'.
    decimal_places : int
        Rounding precision for calorie outputs (default: 2).

    Returns
    -------
    Dict[str, Any]
        Deterministic result containing BMR, TDEE, adjustment, and daily calorie target.
    """
    validated = validate_profile_inputs(
        age_years=age_years,
        sex=sex,
        height_cm=height_cm,
        weight_kg=weight_kg,
        activity_level=activity_level,
        goal=goal,
    )

    bmr = calculate_bmr(
        weight_kg=validated["weight_kg"],
        height_cm=validated["height_cm"],
        age_years=validated["age_years"],
        sex=validated["sex"],
    )

    activity_mult = ACTIVITY_MULTIPLIERS[validated["activity_level"]]
    tdee = bmr * activity_mult
    adjustment = GOAL_ADJUSTMENTS[validated["goal"]]
    daily_target = tdee + adjustment

    return {
        "bmr_kcal": round(bmr, decimal_places),
        "activity_level": validated["activity_level"],
        "activity_multiplier": activity_mult,
        "tdee_kcal": round(tdee, decimal_places),
        "goal": validated["goal"],
        "calorie_adjustment_kcal": adjustment,
        "daily_calorie_target_kcal": round(daily_target, decimal_places),
        "profile": validated,
    }


def save_user_profile(
    profile_data: Dict[str, Any],
    profile_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate, compute targets, and persist user profile to local JSON.

    Parameters
    ----------
    profile_data : Dict[str, Any]
        Dictionary with age_years, sex, height_cm, weight_kg, activity_level, goal.
    profile_path : Optional[Path]
        Target JSON path (default: config.USER_PROFILE_PATH).

    Returns
    -------
    Dict[str, Any]
        Persisted profile payload including calculated targets and timestamp.
    """
    targets = calculate_user_targets(
        age_years=profile_data["age_years"],
        sex=profile_data["sex"],
        height_cm=profile_data["height_cm"],
        weight_kg=profile_data["weight_kg"],
        activity_level=profile_data["activity_level"],
        goal=profile_data["goal"],
    )

    payload = {
        "updated_at": datetime.now().isoformat(),
        "profile": targets["profile"],
        "targets": {
            "bmr_kcal": targets["bmr_kcal"],
            "activity_level": targets["activity_level"],
            "activity_multiplier": targets["activity_multiplier"],
            "tdee_kcal": targets["tdee_kcal"],
            "goal": targets["goal"],
            "calorie_adjustment_kcal": targets["calorie_adjustment_kcal"],
            "daily_calorie_target_kcal": targets["daily_calorie_target_kcal"],
        },
    }

    target_file = Path(profile_path) if profile_path is not None else USER_PROFILE_PATH
    target_file.parent.mkdir(parents=True, exist_ok=True)

    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def load_user_profile(profile_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load user profile and precomputed targets from local JSON.

    Parameters
    ----------
    profile_path : Optional[Path]
        Target JSON path (default: config.USER_PROFILE_PATH).

    Returns
    -------
    Optional[Dict[str, Any]]
        Stored profile payload, or None if file does not exist.
    """
    target_file = Path(profile_path) if profile_path is not None else USER_PROFILE_PATH

    if not target_file.exists():
        return None

    with open(target_file, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_calorie_comparison(
    consumed_calories: float,
    daily_target_calories: Optional[float],
) -> Dict[str, Any]:
    """Calculate remaining calories, percentage, and target comparison status.

    Parameters
    ----------
    consumed_calories : float
        Total calories consumed for the selected date.
    daily_target_calories : Optional[float]
        Configured daily calorie target, or None if no user profile is saved.

    Returns
    -------
    Dict[str, Any]
        Structured comparison metrics including remaining calories, percent of target,
        status ('under', 'met', 'exceeded', or 'no_target'), and descriptive message.
    """
    consumed_rounded = round(float(consumed_calories), 2)

    if daily_target_calories is None:
        return {
            "has_target": False,
            "target_calories": None,
            "consumed_calories": consumed_rounded,
            "remaining_calories": None,
            "percent_of_target": None,
            "status": "no_target",
            "message": "No user profile found. Please save your profile and daily targets to enable comparison.",
        }

    target_rounded = round(float(daily_target_calories), 2)
    remaining = round(target_rounded - consumed_rounded, 2)
    percent = (
        round((consumed_rounded / target_rounded) * 100, 2)
        if target_rounded > 0
        else 0.0
    )

    if remaining < 0:
        status = "exceeded"
        message = f"Daily target exceeded by {abs(remaining):.2f} kcal."
    elif remaining == 0:
        status = "met"
        message = "Daily calorie target met exactly."
    else:
        status = "under"
        message = f"{remaining:.2f} kcal remaining under target."

    return {
        "has_target": True,
        "target_calories": target_rounded,
        "consumed_calories": consumed_rounded,
        "remaining_calories": remaining,
        "percent_of_target": percent,
        "status": status,
        "message": message,
    }


def get_daily_target_comparison(
    date_val: Union[str, dt_date, datetime],
    profile_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compare daily calorie intake against the user's daily calorie target.

    Parameters
    ----------
    date_val : Union[str, dt_date, datetime]
        Target calendar date.
    profile_path : Optional[Path]
        Optional custom user profile JSON path.
    log_path : Optional[Path]
        Optional custom meal log CSV path.

    Returns
    -------
    Dict[str, Any]
        Comparison summary with target, consumed, remaining, and percentage.
    """
    profile_payload = load_user_profile(profile_path=profile_path)
    if profile_payload is None:
        raise ValueError("No user profile found. Please save a user profile before comparing targets.")

    daily_target = float(profile_payload["targets"]["daily_calorie_target_kcal"])
    summary = get_daily_nutrition_summary(date_val, log_path=log_path)
    consumed = float(summary["total_calories"])
    comp = compute_calorie_comparison(consumed, daily_target)

    return {
        "date": summary["date"],
        "target_calories_kcal": comp["target_calories"],
        "consumed_calories_kcal": comp["consumed_calories"],
        "remaining_calories_kcal": comp["remaining_calories"],
        "percent_of_target": comp["percent_of_target"],
        "status": comp["status"],
        "message": comp["message"],
        "meal_count": summary["meal_count"],
        "goal": profile_payload["targets"]["goal"],
        "summary": summary,
    }

