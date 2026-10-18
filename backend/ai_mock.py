"""Mock AI inference for StoolCheck AI (Phase 1 placeholder).

Contract (kept stable so the real YOLOv8/MobileNet model can drop in later):
    analyze_stool_image(image_path: str) -> {
        "bristol_type": int (1-7),
        "label": str,
        "description": str,
        "recommendations": str,
        "confidence": float,
        "model": str,
    }
"""

import random

# Bristol Stool Scale reference data + dietary recommendations per type.
BRISTOL_SCALE: dict[int, dict[str, str]] = {
    1: {
        "label": "Type 1: Separate hard lumps",
        "description": "Hard lumps, like nuts (hard to pass).",
        "recommendation": (
            "Suggests constipation. Increase water intake (8+ glasses/day), "
            "add soluble fiber gradually (oats, beans, fruits), and stay "
            "physically active. Avoid straining."
        ),
    },
    2: {
        "label": "Type 2: Sausage-shaped but lumpy",
        "description": "Like a sausage but with lumps on it.",
        "recommendation": (
            "Mild constipation. Increase fluids and fiber gradually "
            "(whole grains, vegetables, prunes), and maintain a regular "
            "toileting schedule after meals."
        ),
    },
    3: {
        "label": "Type 3: Sausage with cracks",
        "description": "Like a sausage but with cracks on the surface.",
        "recommendation": (
            "Normal/healthy. Keep up your current water, fiber, and activity "
            "levels. This is the ideal consistency."
        ),
    },
    4: {
        "label": "Type 4: Smooth and soft",
        "description": "Like a sausage or snake, smooth and soft.",
        "recommendation": (
            "Ideal Bristol type. Continue a balanced diet rich in fiber, "
            "fluids, and regular exercise to maintain digestive health."
        ),
    },
    5: {
        "label": "Type 5: Soft blobs",
        "description": "Soft blobs with clear-cut edges.",
        "recommendation": (
            "Slightly low fiber. Add more whole grains, vegetables, and legumes. "
            "If accompanied by urgency, reduce fatty or spicy foods."
        ),
    },
    6: {
        "label": "Type 6: Mushy pieces",
        "description": "Fluffy pieces with ragged edges, a mushy stool.",
        "recommendation": (
            "Suggests mild diarrhea or food intolerance. Drink plenty of "
            "fluids with electrolytes, avoid greasy/spicy foods, caffeine, "
            "and dairy if intolerant. Consider a BRAT diet (bananas, rice, "
            "applesauce, toast)."
        ),
    },
    7: {
        "label": "Type 7: Watery, no solids",
        "description": "Entirely liquid, no solid pieces.",
        "recommendation": (
            "Diarrhea. Rehydrate aggressively (water + oral rehydration "
            "solutions), eat bland foods, and avoid lactose, alcohol, and "
            "caffeine. SEEK MEDICAL ATTENTION if it persists over 48 hours, "
            "or if fever/blood is present."
        ),
    },
}


def analyze_stool_image(image_path: str) -> dict:
    """MOCK inference: randomly pick a Bristol type (1-7) and attach advice.

    Replace the body of this function with the real model later; the
    signature and return shape must stay the same.
    """
    bristol_type = random.randint(1, 7)
    entry = BRISTOL_SCALE[bristol_type]
    return {
        "bristol_type": bristol_type,
        "label": entry["label"],
        "description": entry["description"],
        "recommendations": entry["recommendation"],
        "confidence": round(random.uniform(0.75, 0.99), 2),  # fake confidence
        "model": "mock-v1",
    }
