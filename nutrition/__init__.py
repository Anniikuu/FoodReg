"""Nutrition calculation, daily tracking, target management, and recommendation package."""

from nutrition.calculator import (
    calculate_nutrition,
    get_food_nutrition_record,
    load_nutrition_database,
)
from nutrition.recommendations import (
    CLOSE_TO_TARGET_THRESHOLD_KCAL,
    MAX_FOOD_SUGGESTIONS,
    MAX_SUGGESTIONS,
    generate_nutrition_recommendation,
    get_daily_recommendation,
    get_food_suggestions,
)
from nutrition.targets import (
    ACTIVITY_MULTIPLIERS,
    GOAL_ADJUSTMENTS,
    calculate_bmr,
    calculate_user_targets,
    compute_calorie_comparison,
    get_daily_target_comparison,
    load_user_profile,
    save_user_profile,
    validate_profile_inputs,
)
from nutrition.tracker import (
    add_meal_record,
    get_all_meal_records,
    get_daily_nutrition_summary,
    get_meal_records_for_date,
)

__all__ = [
    "calculate_nutrition",
    "get_food_nutrition_record",
    "load_nutrition_database",
    "add_meal_record",
    "get_all_meal_records",
    "get_daily_nutrition_summary",
    "get_meal_records_for_date",
    "calculate_bmr",
    "calculate_user_targets",
    "validate_profile_inputs",
    "save_user_profile",
    "load_user_profile",
    "get_daily_target_comparison",
    "compute_calorie_comparison",
    "generate_nutrition_recommendation",
    "get_daily_recommendation",
    "get_food_suggestions",
    "CLOSE_TO_TARGET_THRESHOLD_KCAL",
    "MAX_FOOD_SUGGESTIONS",
    "MAX_SUGGESTIONS",
    "ACTIVITY_MULTIPLIERS",
    "GOAL_ADJUSTMENTS",
]
