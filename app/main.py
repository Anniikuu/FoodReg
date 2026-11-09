"""Streamlit application: AI Food Recognition, Meal Logging & Daily Nutrition Dashboard."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path for direct module execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import streamlit as st

from config.config import (
    CURRENT_PHASE,
    DEVICE,
    MODEL_OUTPUT_PATH,
    NUM_CLASSES,
    PROJECT_NAME,
)
from models.mobilenet import build_mobilenetv2_model, load_checkpoint, predict_image
from nutrition.calculator import calculate_nutrition
from nutrition.recommendations import generate_nutrition_recommendation
from nutrition.targets import compute_calorie_comparison, load_user_profile
from nutrition.tracker import add_meal_record, get_daily_nutrition_summary


@st.cache_resource
def get_food_classifier():
    """Load and cache the trained MobileNetV2 food classifier."""
    model = build_mobilenetv2_model(num_classes=NUM_CLASSES)
    if MODEL_OUTPUT_PATH.exists():
        load_checkpoint(MODEL_OUTPUT_PATH, model=model, device=DEVICE)
    model.eval()
    return model


def render_food_recognition_section(
    selected_date: Optional[datetime.date] = None,
    log_path: Optional[Path] = None,
) -> None:
    """Render the Image Upload, Food Recognition, and Meal Logging section."""
    st.header("Food Recognition & Meal Logging")

    uploaded_file = st.file_uploader(
        "Upload a food image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload an image of a food item to recognize and calculate its nutrition.",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read image: {e}")
            return

        try:
            model = get_food_classifier()
            with st.spinner("Recognizing food..."):
                pred = predict_image(model, image, device=DEVICE)
        except Exception as e:
            st.error(f"Inference error: {e}")
            return

        cname = pred["predicted_canonical_class"]
        confidence_pct = pred["confidence"] * 100.0
        display_name = cname.replace("_", " ").title()

        st.markdown(f"**Predicted Food:** {display_name} (`{cname}`)")
        st.markdown(f"**Confidence:** {confidence_pct:.1f}%")

        quantity_g = st.number_input(
            "Quantity in grams (g)",
            min_value=1.0,
            max_value=2000.0,
            value=100.0,
            step=10.0,
            help="Enter the portion size in grams.",
        )

        try:
            nutrition = calculate_nutrition(cname, quantity_g)
            if nutrition.get("is_available"):
                st.markdown("#### Estimated Nutrition")
                n1, n2, n3, n4, n5 = st.columns(5)
                n1.metric("Calories", f"{nutrition['calories']:.1f} kcal")
                n2.metric("Protein", f"{nutrition['protein_g']:.1f} g")
                n3.metric("Carbs", f"{nutrition['carbs_g']:.1f} g")
                n4.metric("Fat", f"{nutrition['fat_g']:.1f} g")
                n5.metric("Fiber", f"{nutrition['fiber_g']:.1f} g")
            else:
                st.warning(
                    f"⚠️ Nutrition data for '{display_name}' is currently unresolved in the database. "
                    "This meal cannot be logged."
                )
        except Exception as e:
            st.error(f"Calculation error: {e}")
            return

        log_btn = st.button("Log Meal")
        if log_btn:
            meal_date = selected_date if selected_date is not None else datetime.now().date()
            res = add_meal_record(
                food_identifier=cname,
                quantity_g=quantity_g,
                date=meal_date,
                log_path=log_path,
            )
            if res.get("success"):
                st.info(f"✅ {res['message']}")
            else:
                st.warning(f"⚠️ {res['message']}")


def render_daily_nutrition_dashboard(
    date_val: Optional[datetime.date] = None,
    profile_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> None:
    """Render the Phase 6 Daily Nutrition Dashboard.

    Displays:
      1. Date selector (defaulting to current calendar date)
      2. Calories: Consumed | Target | Remaining (with target exceeded warnings)
      3. Macronutrients: Protein, Carbs, Fat, Fiber
      4. Recorded Meals table for the selected date
      5. Personalized recommendation guidance and suggestions
    """
    st.header("Daily Nutrition Dashboard")

    # 1. Date selection
    if date_val is None:
        selected_date = st.date_input("Select Date", value=datetime.now().date())
    else:
        selected_date = date_val

    # 2. Retrieve daily nutrition totals and stored meal records
    summary = get_daily_nutrition_summary(selected_date, log_path=log_path)

    # 3. Load user profile for daily calorie target
    profile_payload = load_user_profile(profile_path=profile_path)
    daily_target = (
        float(profile_payload["targets"]["daily_calorie_target_kcal"])
        if profile_payload and "targets" in profile_payload
        else None
    )

    # 4. Compute calorie target comparison
    comparison = compute_calorie_comparison(
        consumed_calories=summary["total_calories"],
        daily_target_calories=daily_target,
    )

    # 5. Calories Comparison Section
    st.markdown("### Calories")
    col1, col2, col3 = st.columns(3)
    col1.metric("Consumed", f"{comparison['consumed_calories']:.2f} kcal")
    if comparison["has_target"]:
        col2.metric("Daily Target", f"{comparison['target_calories']:.2f} kcal")
        col3.metric("Remaining", f"{comparison['remaining_calories']:.2f} kcal")
    else:
        col2.metric("Daily Target", "Not set")
        col3.metric("Remaining", "N/A")

    # Status / Alert
    if not comparison["has_target"]:
        st.info("ℹ️ " + comparison["message"])
    elif comparison["status"] == "exceeded":
        st.warning("⚠️ " + comparison["message"])
    elif comparison["status"] == "met":
        st.info("🎯 " + comparison["message"])
    else:
        st.info("✅ " + comparison["message"])

    # 6. Macronutrients Section
    st.markdown("### Today's Nutrition")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Protein", f"{summary['total_protein_g']:.2f} g")
    m2.metric("Carbohydrates", f"{summary['total_carbs_g']:.2f} g")
    m3.metric("Fat", f"{summary['total_fat_g']:.2f} g")
    m4.metric("Fiber", f"{summary['total_fiber_g']:.2f} g")

    # 7. Recorded Meals List / Table
    st.markdown("### Recorded Meals")
    meals = summary["meals"]
    if not meals:
        st.info("No meals recorded for this date.")
    else:
        table_rows = []
        for m in meals:
            time_str = m["timestamp"].split(" ")[-1] if " " in m["timestamp"] else m["timestamp"]
            table_rows.append({
                "Food": m["food_name"],
                "Quantity (g)": m["quantity_g"],
                "Calories (kcal)": m["calories_kcal"],
                "Protein (g)": m["protein_g"],
                "Carbs (g)": m["carbs_g"],
                "Fat (g)": m["fat_g"],
                "Fiber (g)": m["fiber_g"],
                "Time": time_str,
            })
        st.dataframe(table_rows, use_container_width=True)

    # 8. Personalized Recommendation Section
    st.markdown("### Personalized Recommendation")
    rec = generate_nutrition_recommendation(
        consumed_calories=summary["total_calories"],
        daily_target_calories=daily_target,
        protein_g=summary["total_protein_g"],
        meal_count=summary["meal_count"],
    )

    if rec["status"] == "target_exceeded":
        st.warning(f"⚠️ **{rec['title']}**: {rec['message']}")
    elif rec["status"] == "target_met":
        st.info(f"🎯 **{rec['title']}**: {rec['message']}")
    else:
        st.info(f"ℹ️ **{rec['title']}**: {rec['message']}")

    st.caption(f"ℹ️ {rec['protein_guidance']}")

    # 9. Suggested Foods Section (Step 6)
    suggestions = rec.get("suggestions", [])
    if suggestions:
        st.markdown("### Suggested Foods")
        st.caption(rec.get("suggestions_explanation", ""))
        suggestion_rows = []
        for s in suggestions:
            suggestion_rows.append({
                "Food": s["food_name"],
                "Calories / 100g": f"{s['calories_kcal_100g']:.1f}",
                "Protein / 100g": f"{s['protein_g_100g']:.2f}",
            })
        st.dataframe(suggestion_rows, use_container_width=True)


def main() -> None:
    """Run Streamlit application."""
    st.set_page_config(page_title="AI Food Recog & Nutrition Tracking", layout="centered")
    st.title(PROJECT_NAME)
    st.subheader(CURRENT_PHASE)
    st.success("Phase 1 setup is complete. Application health check passed.")

    st.divider()
    render_food_recognition_section()

    st.divider()
    render_daily_nutrition_dashboard()


if __name__ == "__main__":
    main()
