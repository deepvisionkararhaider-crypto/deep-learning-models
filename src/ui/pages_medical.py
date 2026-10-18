"""Medical AI — brain-MRI classification, segmentation, detection, instance segmentation."""
from __future__ import annotations

import streamlit as st

from ..registry import MEDICAL
from ..services import manifest_service as MS
from .common import hero, inject_css, medical_disclaimer
from .predict import studio


def render() -> None:
    inject_css()
    hero("Medical AI — Brain Tumour Analysis",
         "Upload a brain-MRI slice and run one of the verified medical models. Educational and "
         "research use only.", "🩺")
    medical_disclaimer()

    entries = MS.entries()
    ready = [s for s in MEDICAL if entries.get(s.id, {}).get("status") == "TRAINED_VERIFIED"]
    st.caption(f"**{len(ready)} / {len(MEDICAL)}** medical models are ready for inference.")

    st.divider()
    studio(MEDICAL, key_prefix="med", area_label="medical")

    st.divider()
    with st.expander("Responsible use, data and limitations", expanded=False):
        st.markdown(
            "- **Not a medical device.** Outputs are statistical estimates from a research "
            "prototype. They are not a diagnosis, have not been prospectively validated, and must "
            "not drive treatment decisions.\n"
            "- **No patient data is stored.** Uploaded images are processed in the Streamlit "
            "session only and are never written to disk or sent to a third party.\n"
            "- **Datasets are public research corpora** (see `DATASETS.md`). Their scanner, "
            "protocol and population biases are inherited by these models.\n"
            "- **Segmentation is not a substitute for classification**, and vice versa — each task "
            "has its own model and its own output type.\n"
            "- **Splits:** the source dataset's own train/validation/test split is used where it "
            "exists; patient-disjointness is inherited from the source and not independently "
            "re-verified here.\n"
        )
