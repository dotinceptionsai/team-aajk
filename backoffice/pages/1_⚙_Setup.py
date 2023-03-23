import streamlit as st

import backoffice.shared as shared
import extra_streamlit_components as stx


with st.spinner("Loading..."):
    from backoffice.steps import (
        data_selection,
        model_choice,
        check_word_level,
        calibration,
        check_sentence_level,
    )

st.session_state.update(
    {k: v for k, v in st.session_state.items() if k.startswith("SessionKey")}
)

shared.make_sidebar()
datasets = shared.get_ds_registry()


st.write(
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.3.0/css/all.min.css" />',
    unsafe_allow_html=True,
)

current_step = stx.stepper_bar(
    steps=[
        "Import data",
        "Words",
        "Sentences",
        "Select Model",
        "Calibrate",
        "Verify",
    ]
)

# <i class="fa-solid fa-check"></i>
# <i class="fa-solid fa-solid fa-check-double"></i>
# <i class="fa-solid fa-certificate"></i>
# <i class="fa-solid fa-spell-check"></i>fa-gears
# <i class="fa-solid fa-list-check"></i>
# <i class="fa-solid fa-circle-check"></i>

if current_step == 0:
    st.write(
        '<h2><i class="fa-solid fa-cloud-arrow-up" style="font-size:40px;color:purple"></i> Import data </h2>',
        unsafe_allow_html=True,
    )

    data_selection.display()

elif current_step == 1:
    st.write(
        '<h2><i class="fa-solid fa-spell-check" style="font-size:40px;color:purple"></i> Check words</h2>',
        unsafe_allow_html=True,
    )
    check_word_level.display()
elif current_step == 2:
    st.write(
        '<h2><i class="fa-solid fa-list-check" style="font-size:40px;color:purple"></i> Check sentences</h2>',
        unsafe_allow_html=True,
    )
    check_sentence_level.display()
elif current_step == 3:
    st.write(
        '<h2><i class="fa-solid fa-gears" style="font-size:40px;color:purple"></i> Model selection</h2>',
        unsafe_allow_html=True,
    )
    model_choice.display()
elif current_step == 4:
    st.write(
        '<h2><i class="fa-solid fa-bullseye" style="font-size:40px;color:purple"></i> Calibration</h2>',
        unsafe_allow_html=True,
    )
    calibration.display()
elif current_step == 5:
    st.write(
        '<h2><i class="fa-solid fa-certificate" style="font-size:40px;color:purple"></i> Complete</h2>',
        unsafe_allow_html=True,
    )
    from backoffice.steps import finish

    finish.display()
else:
    st.write("Not implemented")
