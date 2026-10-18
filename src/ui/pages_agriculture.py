"""Smart Agriculture AI — leaf-disease classification, segmentation, cotton detection."""
from __future__ import annotations

import streamlit as st

from ..registry import AGRI
from ..services import manifest_service as MS
from .common import agri_disclaimer, hero, inject_css
from .predict import studio

CLASS_NOTES = {
    "healthy": "No disease detected by the model. Healthy leaves should still be monitored.",
    "Apple___Apple_scab": "Apple scab — fungal (Venturia inaequalis); favour resistant cultivars "
                          "and sanitation.",
    "Apple___Black_rot": "Apple black rot — fungal; remove mummified fruit and cankers.",
    "Apple___Cedar_apple_rust": "Cedar-apple rust — needs both hosts; manage nearby junipers.",
    "Corn___Cercospora_leaf_spot": "Gray leaf spot — fungal; rotate crops and use resistant hybrids.",
    "Corn___Common_rust": "Common rust — fungal; scout early, use resistant hybrids.",
    "Corn___Northern_Leaf_Blight": "Northern leaf blight — fungal; residue management helps.",
    "Potato___Early_blight": "Early blight (Alternaria) — fungal; avoid overhead irrigation.",
    "Potato___Late_blight": "Late blight (Phytophthora) — aggressive; act quickly, destroy "
                            "infected plants.",
    "Tomato___Bacterial_spot": "Bacterial spot — copper sprays and clean seed help.",
    "Tomato___Early_blight": "Early blight — fungal; stake plants and remove lower leaves.",
    "Tomato___Late_blight": "Late blight — serious; remove and destroy infected plants.",
    "Tomato___Leaf_Mold": "Leaf mold — improve airflow and reduce humidity.",
    "Tomato___Septoria_leaf_spot": "Septoria leaf spot — remove affected leaves, mulch.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Yellow leaf curl virus — whitefly-vectored; manage "
                                              "whiteflies.",
    "Tomato___Tomato_mosaic_virus": "Mosaic virus — viral, no cure; remove plants, disinfect tools.",
}


def render() -> None:
    inject_css()
    hero("Smart Agriculture AI — Plant Disease Analysis",
         "Leaf-disease classification (25 classes), leaf-disease segmentation and cotton disease "
         "detection, using verified checkpoints.", "🌿")
    agri_disclaimer()

    entries = MS.entries()
    ready = [s for s in AGRI if entries.get(s.id, {}).get("status") == "TRAINED_VERIFIED"]
    st.caption(f"**{len(ready)} / {len(AGRI)}** agriculture models are ready for inference.")

    st.divider()
    studio(AGRI, key_prefix="agri", area_label="agriculture")

    st.divider()
    with st.expander("Disease notes for the 25 plant classes", expanded=False):
        st.caption("These notes describe the diseases the classifier was trained on. They are "
                   "general guidance, not a treatment prescription.")
        for label in sorted(CLASS_NOTES):
            st.markdown(f"- **{label.replace('___', ' · ')}** — {CLASS_NOTES[label]}")

    with st.expander("Field-use limitations", expanded=False):
        st.markdown(
            "- **Domain shift is real.** The training images were captured under specific "
            "conditions; performance on a different device, lighting or growth stage is not "
            "guaranteed.\n"
            "- **Detection coverage is narrow.** The cotton detector was trained on boxes that are "
            "only a few pixels wide in 640×640 images, so a single-scale detector has very low "
            "recall — this is documented, not hidden.\n"
            "- **Instance segmentation for agriculture is BLOCKED** — no instance-level dataset "
            "with a usable parquet mirror exists, so no model is offered rather than faking masks.\n"
            "- Verify economically important decisions with an agronomist or extension service.\n"
        )
